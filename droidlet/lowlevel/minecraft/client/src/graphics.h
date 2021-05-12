// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <cmath>
#include <vector>

#include "block_map.h"
#include "types.h"

class Graphics {
 public:
  // Calculate block/depth vision from player at position `pos` in direction `look`
  //
  // Arguments:
  // - blocks will be populated with id/meta pairs, and must be (height x width) in length
  // - depth will be populated with distances to the block, and must be (height x width) in length
  // - blockPos will be populated with xyz of blocks seen in the image, and must be
  //   (height x width) in length
  // - height, width are the requested dimensions of the image
  // - blockMap holds the blocks that are to be rendered
  // - pos is the position of the player's feet (rendering is done from their eyes)
  // - look is the direction the camera is pointing
  // - maxDepth is the number of blocks a client can see in any direction
  // - moveToPlayerHead whether the camera should be set at the player head or not
  //   For agent client rendering, this is true; for offline house rendering, to
  //   match Chunky's setting, this should be false.
  static void vision(std::vector<Block>& blocks, std::vector<float>& depth,
                     std::vector<BlockPos>& blockPos, int height, int width,
                     const BlockMap& blockMap, Pos pos, Look look, int maxDepth,
                     bool moveToPlayerHead = true);

  // Find the closest block in a player's direct line of sight
  // - pos is the position of the player's feet (rendering is done from their eyes)
  // - look is the direction the camera is pointing
  // - maxDepth is the number of blocks a client can see in any direction
  //
  // Returns true iff there was a block in the line of sight, and sets *block, *depth, and *blockPos
  static bool lineOfSight(Block* block, float* depth, BlockPos* blockPos, const BlockMap& blockMap,
                          Pos pos, Look look, int maxDepth);

  // Convert a Look (yaw/pitch angles) to a 3d unit vector
  static Pos toUnitVec(Look look);

 private:
  // Magnitude of 3d vector
  static float magnitude(Pos v);

  // In the pinhole projection setting, get the X or Y for a pixel
  // (0 <= pix <= halfSize * 2) given Z==1
  // Whether it returns X or Y depends on whether halfSize is width/2 or height/2
  // pix is the pixel coordinate on the image plane
  // The X or Y is computed on a plane of Z==1 and the camera has Z==0.
  static float getXY(int pix, int halfSize, float halfFov);

  // Set *block, *depth, and *blockPos to the first ray collision starting from `eye`
  // pointing towards `ray`
  //
  // Returns true iff there was an intersection
  static bool setPixel(Block* block, float* depth, BlockPos* blockPos, const BlockMap& blockMap,
                       Pos eye, Pos ray, int renderDistance);

  // Find the block at a coordinate/plane intersection.
  //
  // Arguments:
  // - p is the intersection point
  // - dim is the dimension whose plane `p` intersects
  // - dw is the plane iterator increment, -1 if looking in neg-dim direction, +1 if pos-dim
  // - blockMap holds the blocks being rendered
  static std::tuple<BlockPos, Block> blockAtPoint(Pos p, int dim, int dw, const BlockMap& blockMap);
};
