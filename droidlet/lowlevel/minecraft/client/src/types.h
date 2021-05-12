// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <glog/logging.h>
#include <math.h>
#include <array>
#include <memory>
#include <string>

struct Pos;

enum GameMode {
  CREATIVE,
  SURVIVAL,
};

enum WindowType {
  PLAYER_INVENTORY,
  CRAFTING_TABLE,
};

struct BlockPos {
  int x, y, z;

  int operator[](int x) {
    switch (x) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        LOG(FATAL) << "Invalid BlockPos index " << x;
    }
  }

  Pos toPos();
  Pos center();

  bool operator==(const BlockPos& rhs) const;
  bool operator!=(const BlockPos& rhs) const;
};

struct Pos {
  double x, y, z;

  BlockPos toBlockPos() {
    return {
        static_cast<int>(floor(x)), static_cast<int>(floor(y)), static_cast<int>(floor(z)),
    };
  }

  double& operator[](const int idx) {
    switch (idx) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        LOG(FATAL) << "Invalid Pos index " << x;
    }
  }
};

struct Look {
  float yaw, pitch;
};

// A single block, which has an 8-bit id and 4-bit meta
struct Block {
  uint8_t id, meta;

  bool operator==(const Block& b) const;
  bool operator!=(const Block& b) const;
};

struct BlockWithPos {
  BlockPos pos;
  Block block;
};

// 16x16x16 blocks, ordered (y, z, x)
typedef std::shared_ptr<std::array<Block, 4096>> ChunkSectionBlocks;

struct ChunkSection {
  // Chunk index
  int cx, cy, cz;

  // 16x16x16 blocks, ordered (y, z, x)
  ChunkSectionBlocks blocks;
};

struct Item {
  uint16_t id;
  uint8_t meta;

  bool operator==(const Item& x) const { return id == x.id && meta == x.meta; }
};

struct Slot {
  uint16_t id;
  uint8_t meta;
  uint8_t count;
  uint16_t damage;

  bool empty() const { return id == 0; }
};

const Slot EMPTY_SLOT = {0, 0, 0, 0};

struct Player {
  std::string uuid;
  std::string name;
  uint64_t entityId;
  float health;
  Pos pos;
  Look look;
  uint32_t foodLevel;
  Slot mainHand;
};

struct Mob {
  std::string uuid;
  uint64_t entityId;
  uint8_t mobType;
  Pos pos;
  Look look;
};

struct Object {
  std::string uuid;
  uint64_t entityId;
  uint8_t objectType;
  Pos pos;
};

struct ItemStack {
  std::string uuid;
  uint64_t entityId;
  Slot item;
  Pos pos;
};

////////////////
// Print operators

std::ostream& operator<<(std::ostream& os, const Pos& pos);
std::ostream& operator<<(std::ostream& os, const BlockPos& pos);
std::ostream& operator<<(std::ostream& os, const Block& b);
std::ostream& operator<<(std::ostream& os, const Look& look);
std::ostream& operator<<(std::ostream& os, const Slot& slot);

////////////////
// Arithmetic operators

Pos operator+(const Pos& a, const Pos& b);
Pos operator-(const Pos& a, const Pos& b);
Pos operator+(const Pos& a, const BlockPos& b);
Pos operator-(const Pos& a, const BlockPos& b);
Pos operator+(const Pos& a, double d);
Pos operator-(const Pos& a, double d);
Pos operator*(const Pos& a, double d);
BlockPos operator+(const BlockPos& a, const BlockPos& b);
BlockPos operator-(const BlockPos& a, const BlockPos& b);
BlockPos operator*(const BlockPos& a, int c);

////////////////
// Hash functions

namespace std {
template <>
struct hash<Item> {
  size_t operator()(const Item& x) const { return ((size_t)x.id << 16) | x.meta; }
};
}  // namespace std
