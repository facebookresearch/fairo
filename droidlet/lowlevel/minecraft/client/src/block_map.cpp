// Copyright (c) Facebook, Inc. and its affiliates.

#include <glog/logging.h>
#include <boost/functional/hash.hpp>
#include <unordered_map>

#include "block_data.h"
#include "block_map.h"
#include "event.h"
#include "types.h"
#include "util.h"

using namespace std;
using std::optional;

// Chunks are 16x16x16 cubes of blocks
const int CHUNK_EDGE = 16;
const int BLOCKS_PER_CHUNK = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;

////////////////
// Public

void BlockMap::setChunk(ChunkSection chunk) {
  lock_.lock();
  setChunkUnsafe(chunk);
  lock_.unlock();
}

bool BlockMap::isChunkLoaded(int cx, int cy, int cz) {
  lock_.lock();
  optional<ChunkSection> chunk = getChunkUnsafe(cx, cy, cz);
  lock_.unlock();
  return (bool)chunk;
}

bool BlockMap::isBlockLoaded(BlockPos p) {
  int cx = pyDivmod(p.x, CHUNK_EDGE).first;
  int cy = pyDivmod(p.y, CHUNK_EDGE).first;
  int cz = pyDivmod(p.z, CHUNK_EDGE).first;
  return isChunkLoaded(cx, cy, cz);
}

bool BlockMap::chunkExists(int cx, int cy, int cz) {
  lock_.lock();
  optional<ChunkSection> c = getChunkUnsafe(cx, cy, cz);
  lock_.unlock();
  return (bool)c;
}

void BlockMap::setBlock(BlockPos pos, Block block) {
  auto cx_ox = pyDivmod(pos.x, CHUNK_EDGE);
  auto cy_oy = pyDivmod(pos.y, CHUNK_EDGE);
  auto cz_oz = pyDivmod(pos.z, CHUNK_EDGE);
  lock_.lock();
  ChunkSection chunk = getChunkOrThrowUnsafe(cx_ox.first, cy_oy.first, cz_oz.first);
  if (chunk.blocks == NULL) {
    // There is a block change to a chunk that was previously all-air. Generate
    // an all-air chunk to modify.
    chunk.blocks = make_shared<array<Block, BLOCKS_PER_CHUNK>>();
    setChunkUnsafe(chunk);
  }
  (*chunk.blocks)[index(cx_ox.second, cy_oy.second, cz_oz.second)] = block;
  lock_.unlock();
}

optional<Block> BlockMap::getBlockUnsafe(int x, int y, int z) const {
  auto cx_ox = pyDivmod(x, CHUNK_EDGE);
  auto cy_oy = pyDivmod(y, CHUNK_EDGE);
  auto cz_oz = pyDivmod(z, CHUNK_EDGE);
  optional<ChunkSection> chunk = getChunkUnsafe(cx_ox.first, cy_oy.first, cz_oz.first);
  if (!chunk) {
    return optional<Block>{};
  }
  if (chunk->blocks == NULL) {
    return BLOCK_AIR;
  }
  return (*chunk->blocks)[index(cx_ox.second, cy_oy.second, cz_oz.second)];
}

optional<Block> BlockMap::getBlock(int x, int y, int z) const {
  lock_.lock();
  auto b = getBlockUnsafe(x, y, z);
  lock_.unlock();
  return b;
}

Block BlockMap::getBlockOrThrow(int x, int y, int z) const {
  optional<Block> block = getBlock(x, y, z);
  CHECK(block) << "Block (" << x << ", " << y << ", " << z << ") does not exist";
  return *block;
}

bool BlockMap::canWalkthrough(int x, int y, int z) const {
  optional<Block> b = getBlock(x, y, z);
  return b && WALKTHROUGH_BLOCKS.count(b->id) > 0;
}

bool BlockMap::canStandAt(int x, int y, int z) const {
  return canWalkthrough(x, y, z) && !canWalkthrough(x, y - 1, z);
}

void BlockMap::getCuboid(vector<Block>& ob, int xa, int xb, int ya, int yb, int za, int zb) {
  int xspan = (xb - xa + 1);
  int yspan = (yb - ya + 1);
  int zspan = (zb - za + 1);
  CHECK_EQ(xspan * yspan * zspan, ob.size());

  auto xa_dm = pyDivmod(xa, CHUNK_EDGE);
  auto xb_dm = pyDivmod(xb, CHUNK_EDGE);
  auto ya_dm = pyDivmod(ya, CHUNK_EDGE);
  auto yb_dm = pyDivmod(yb, CHUNK_EDGE);
  auto za_dm = pyDivmod(za, CHUNK_EDGE);
  auto zb_dm = pyDivmod(zb, CHUNK_EDGE);

  lock_.lock();

  for (int cx = xa_dm.first; cx <= xb_dm.first; cx++) {
    for (int cy = ya_dm.first; cy <= yb_dm.first; cy++) {
      for (int cz = za_dm.first; cz <= zb_dm.first; cz++) {
        optional<ChunkSection> chunk = getChunkUnsafe(cx, cy, cz);
        if (!chunk || chunk->blocks == NULL) {
          continue;
        }

        int ox_min = (cx == xa_dm.first) ? xa_dm.second : 0;
        int ox_max = (cx == xb_dm.first) ? xb_dm.second : (CHUNK_EDGE - 1);
        int oy_min = (cy == ya_dm.first) ? ya_dm.second : 0;
        int oy_max = (cy == yb_dm.first) ? yb_dm.second : (CHUNK_EDGE - 1);
        int oz_min = (cz == za_dm.first) ? za_dm.second : 0;
        int oz_max = (cz == zb_dm.first) ? zb_dm.second : (CHUNK_EDGE - 1);

        // Copy all blocks in a chunk in one go
        for (int oy = oy_min; oy <= oy_max; oy++) {
          for (int oz = oz_min; oz <= oz_max; oz++) {
            for (int ox = ox_min; ox <= ox_max; ox++) {
              Block b = (*chunk->blocks)[index(ox, oy, oz)];
              auto idx = (cy * CHUNK_EDGE + oy - ya) * zspan * xspan +
                         (cz * CHUNK_EDGE + oz - za) * xspan + (cx * CHUNK_EDGE + ox - xa);

              ob[idx] = b;
            }
          }
        }
      }
    }
  }

  lock_.unlock();
}

////////////////
// Private

optional<ChunkSection> BlockMap::getChunkUnsafe(int cx, int cy, int cz) const {
  auto chunk = chunks_.find(key(cx, cy, cz));
  if (chunk != chunks_.end()) {
    return chunk->second;
  } else {
    return optional<ChunkSection>{};
  }
}

ChunkSection BlockMap::getChunkOrThrowUnsafe(int cx, int cy, int cz) const {
  optional<ChunkSection> chunk = getChunkUnsafe(cx, cy, cz);
  CHECK(chunk) << "Chunk (" << cx << ", " << cy << ", " << cz << ") does not exist";
  return *chunk;
}

void BlockMap::setChunkUnsafe(ChunkSection chunk) {
  chunks_[key(chunk.cx, chunk.cy, chunk.cz)] = chunk;
}

size_t BlockMap::key(int x, int y, int z) {
  size_t seed = 0;
  boost::hash_combine(seed, x);
  boost::hash_combine(seed, y);
  boost::hash_combine(seed, z);
  return seed;
}

inline int BlockMap::index(uint8_t ox, uint8_t oy, uint8_t oz) {
  return (oy << 8) + (oz << 4) + ox;
}
