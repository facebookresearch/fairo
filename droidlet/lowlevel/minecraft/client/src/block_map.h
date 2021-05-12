// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <boost/functional/hash.hpp>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "event.h"
#include "types.h"

// BlockMap is a subset of the GameState which handles the chunk map. Generally,
// it is a map of (x, y, z) -> (block id). Practically, it is stored as a map of
// (chunk x/y/z) -> (chunk), where chunk is a 16x16x16 array of blocks ordered
// (y, z, x).
//
// This is the way the Minecraft server/protocol work, so it is convenient to
// keep that structure here.
//
// THREAD SAFETY
//
// All public methods are thread-safe unless suffixed with "Unsafe". Unsafe
// methods can be safely called by wrapping in public methods lock() and
// unlock().
//
class BlockMap {
 public:
  // Thread-safety
  void lock() const { lock_.lock(); }
  void unlock() const { lock_.unlock(); }

  // Chunk management
  void setChunk(ChunkSection chunk);
  bool isChunkLoaded(int cx, int cy, int cz);
  bool chunkExists(int cx, int cy, int cz);

  // Block management
  // Use getBlockOrThrow() if you are certain the block exists
  bool isBlockLoaded(BlockPos p);
  void setBlock(BlockPos pos, Block block);
  std::optional<Block> getBlockUnsafe(int x, int y, int z) const;
  std::optional<Block> getBlock(int x, int y, int z) const;
  std::optional<Block> getBlock(BlockPos p) const { return getBlock(p.x, p.y, p.z); }
  Block getBlockOrThrow(int x, int y, int z) const;
  Block getBlockOrThrow(BlockPos p) const { return getBlockOrThrow(p.x, p.y, p.z); }

  // Returns true iff the block at (x, y, z) is not solid (i.e. returns
  // true for air and flowers, false for dirt and stone)
  bool canWalkthrough(int x, int y, int z) const;
  bool canWalkthrough(BlockPos p) { return canWalkthrough(p.x, p.y, p.z); }
  bool canWalkthrough(Pos p) { return canWalkthrough(p.toBlockPos()); }

  // Returns true if the player is allowed to stand at the given position.
  // The block must be not solid, and the block below must be solid.
  bool canStandAt(int x, int y, int z) const;
  bool canStandAt(BlockPos p) const { return canStandAt(p.x, p.y, p.z); }
  bool canStandAt(Pos p) const { return canStandAt(p.toBlockPos()); }

  // Fill ob with the cuboid bounded inclusively by corners (xa, ya, za), (xb, yb, zb)
  void getCuboid(std::vector<Block>& ob, int xa, int xb, int ya, int yb, int za, int zb);

 private:
  std::optional<ChunkSection> getChunkUnsafe(int cx, int cy, int cz) const;
  ChunkSection getChunkOrThrowUnsafe(int cx, int cy, int cz) const;
  void setChunkUnsafe(ChunkSection chunk);

  // Get map key from 3-tuple
  static size_t key(int x, int y, int z);

  // Get index of block in chunk given offsets (ox, oy, oz).
  // Blocks are ordered (y, z, x).
  inline static int index(uint8_t ox, uint8_t oy, uint8_t oz);

  // Fields
  std::unordered_map<size_t, ChunkSection> chunks_;
  mutable std::mutex lock_;
};
