// Copyright (c) Facebook, Inc. and its affiliates.

#include "graphics.h"
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <thread>

using namespace std;
using Eigen::Vector3f;
using Eigen::AngleAxisf;
using Eigen::Quaternionf;
using std::optional;

// The camera will have a view of FOV degrees in both x and y directions
// 35 (70/2) is chunky's default setting
static const float FOV = 35;

// The camera eye will be this height above the player's feet
// chunky has 1.6 as the height
static const float PLAYER_HEIGHT = 1.6;

// Number of threads to spawn per call
static const int N_THREADS = 8;

// Avoid trying to find intersections in a dimension exactly parallel to the
// view. e.g. if the look unit vector is {0.7071, 0, 0.7071}, which is exactly
// parallel to the ground, then the top/bottom of blocks will not be visible,
// but will cause issues with floating-point math. Avoid this dimension
// entirely.
static const float MIN_RAY_DIM = .0001;

void Graphics::vision(vector<Block>& blocks, vector<float>& depth, vector<BlockPos>& blockPos,
                      int height, int width, const BlockMap& blockMap, Pos pos, Look look,
                      int maxDepth, bool moveToPlayerHead) {
  auto tStart = chrono::high_resolution_clock::now();

  // Clear space in output vectors
  blocks.resize(height * width);
  depth.resize(height * width);
  blockPos.resize(height * width);

  // Calculate from the perspective of player's head, not feet
  Pos eye = pos;
  if (moveToPlayerHead) {
    eye = eye + Pos{0, PLAYER_HEIGHT, 0};
  }

  // TODO: spawn thread pool once, reuse for subsequent calls to vision()
  vector<thread> threads;

  // Start critical section
  blockMap.lock();

  // Eigen's coordinate system
  //      ^ Y
  //      |
  //      |
  //     / ------> X
  //   /
  //  v Z

  // Agent camera's coordinate system
  //       ^ Y
  //       |    ^ Z
  //       |  /
  // X <---|/

  Quaternionf camera = AngleAxisf(look.pitch / 180 * M_PI, Vector3f::UnitX()) *
                       AngleAxisf(look.yaw / 180 * M_PI, Vector3f::UnitY());
  auto camera_m = camera.toRotationMatrix();

  for (int tid = 0; tid < N_THREADS; tid++) {
    threads.push_back(thread(
        [&](int tid) {
          int stride = (height + N_THREADS - 1) / N_THREADS;
          for (int i = tid * stride; i < std::min((tid + 1) * stride, height); i++) {
            for (int j = 0; j < width; j++) {
              int k = (i * width) + j;
              // the calc of x and y follows exactly chunky's PinholeProjector
              float x = getXY(j, width / 2, FOV) * width / height;
              float y = getXY(i, height / 2, FOV);
              // follow Eigen's coordinate system
              auto camera_ray = Vector3f(x, -y, -1.0);
              camera_ray.normalize();
              // transform to the world space
              auto ray = camera_m.inverse() * camera_ray;
              // convert to the Agent camera's coordinate system
              setPixel(&blocks[k], &depth[k], &blockPos[k], blockMap, eye,
                       Pos{-ray(0), ray(1), -ray(2)}, maxDepth);
            }
          }
        },
        tid));
  }
  for (int tid = 0; tid < N_THREADS; tid++) {
    threads[tid].join();
  }

  // End critical section
  blockMap.unlock();

  auto tEnd = chrono::high_resolution_clock::now();
  auto tMicros = chrono::duration_cast<chrono::microseconds>(tEnd - tStart).count();
  if (tMicros > 5000) {
    LOG(INFO) << "Graphics rendered " << height * width << "px in " << tMicros << "us";
  }
}

float Graphics::getXY(int pix, int halfSize, float halfFov) {
  float offset = pix - halfSize;
  halfFov = halfFov / 180 * M_PI;
  return offset / halfSize * tan(halfFov);
}

bool Graphics::lineOfSight(Block* block, float* depth, BlockPos* blockPos, const BlockMap& blockMap,
                           Pos pos, Look look, int maxDepth) {
  // Calculate from the perspective of player's head, not feet
  Pos eye = pos + Pos{0, PLAYER_HEIGHT, 0};

  blockMap.lock();
  Pos ray = toUnitVec(look);
  bool valid = setPixel(block, depth, blockPos, blockMap, eye, ray, maxDepth);
  blockMap.unlock();

  return valid;
}

bool Graphics::setPixel(Block* block, float* depth, BlockPos* blockPos, const BlockMap& blockMap,
                        Pos eye, Pos ray, int maxDepth) {
  *depth = numeric_limits<float>::max();
  *block = {0, 0};
  int farthest = numeric_limits<int>::min();
  *blockPos = {farthest, farthest, farthest};

  // Check each dimension. Look for collisions in the y-plane,
  // z-plane, and x-plane, and choose the closest one
  for (int dim = 0; dim < 3; dim++) {
    if (abs(ray[dim]) < MIN_RAY_DIM) {
      continue;
    }

    // Check closer planes before farther planes so that the first hit is
    // guaranteed to be the closest.
    long w = ray[dim] > 0 ? long(ceil(eye[dim])) : long(floor(eye[dim]));
    long dw = ray[dim] > 0 ? 1 : -1;
    while (true) {
      // Get ray multiple to plane (equal to distance because ray is a unit vector)
      float dist = (w - eye[dim]) / ray[dim];
      CHECK(dist >= 0) << "Plane is behind the camera. eye=" << eye << " ray=" << ray
                       << " dim=" << dim << " w=" << w << " dist=" << dist;

      // If the closest block in this dim is farther than our closest block in
      // another dim, or farther than the render distance, exit this whole dim
      if (dist > maxDepth || dist > *depth) {
        break;
      }

      // Get block at plane intersection
      Pos vecToPlane = ray * dist;
      Pos intersection = eye + vecToPlane;
      tuple<BlockPos, Block> posBlock = blockAtPoint(intersection, dim, dw, blockMap);
      Block b = get<1>(posBlock);

      // If there is a block, it is the closest one we have found yet, and we
      // will not find a closer one in this dim, so break
      if (b.id != 0) {
        *block = b;
        *depth = dist;
        *blockPos = get<0>(posBlock);
        break;
      }

      w += dw;
    }
  }
  return block->id != 0;
}

tuple<BlockPos, Block> Graphics::blockAtPoint(Pos p, int dim, int dw, const BlockMap& blockMap) {
  BlockPos unitDim = {dim == 0, dim == 1, dim == 2};
  BlockPos bp = (p + (unitDim.toPos() * 0.5)).toBlockPos();
  // If looking in the -x direction, you will see blocks on their +x side. A
  // block at x=5 will be seen by a ray intersection at x=6, thus we fetch the
  // block at (x - 1)
  if (dw < 0) {
    bp = bp - unitDim;
  }
  optional<Block> block = blockMap.getBlockUnsafe(bp.x, bp.y, bp.z);
  if (block && block->id != 0) {
    return make_tuple(bp, *block);
  }

  // Else, return air
  return make_tuple(BlockPos{0, 0, 0}, Block{0, 0});
}

Pos Graphics::toUnitVec(Look look) {
  float pitch = look.pitch * M_PI / 180;
  float yaw = look.yaw * M_PI / 180;
  return {
      -1 * cos(pitch) * sin(yaw), -1 * sin(pitch), cos(pitch) * cos(yaw),
  };
}
