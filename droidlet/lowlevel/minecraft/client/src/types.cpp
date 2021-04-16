// Copyright (c) Facebook, Inc. and its affiliates.

#include "types.h"
#include <iostream>
#include "math.h"

using namespace std;

Pos BlockPos::toPos() {
  return {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
}

Pos BlockPos::center() { return toPos() + Pos{0.5, 0.5, 0.5}; }

////////////////
// Print operators

ostream& operator<<(ostream& os, const Pos& pos) {
  return os << "(" << pos.x << ", " << pos.y << ", " << pos.z << ")";
}

ostream& operator<<(ostream& os, const BlockPos& pos) {
  return os << "(" << pos.x << ", " << pos.y << ", " << pos.z << ")";
}

ostream& operator<<(ostream& os, const Block& b) { return os << (int)b.id << ":" << (int)b.meta; }

ostream& operator<<(ostream& os, const Look& look) {
  return os << "(" << look.yaw << ", " << look.pitch << ")";
}
ostream& operator<<(ostream& os, const Slot& slot) {
  return os << "(" << (int)slot.id << ", " << (int)slot.meta << ", " << (int)slot.count << ")";
}

////////////////
// Arithmetic operators

bool Block::operator==(const Block& b) const { return id == b.id && meta == b.meta; }

bool Block::operator!=(const Block& b) const { return id != b.id || meta != b.meta; }

bool BlockPos::operator==(const BlockPos& r) const { return x == r.x && y == r.y && z == r.z; }

bool BlockPos::operator!=(const BlockPos& r) const { return x != r.x || y != r.y || z != r.z; }

Pos operator+(const Pos& a, const Pos& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

Pos operator-(const Pos& a, const Pos& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

Pos operator+(const Pos& a, const BlockPos& b) {
  return {a.x + (double)b.x, a.y + (double)b.y, a.z + (double)b.z};
}

Pos operator-(const Pos& a, const BlockPos& b) {
  return {a.x - (double)b.x, a.y - (double)b.y, a.z - (double)b.z};
}

Pos operator+(const Pos& a, double d) { return {a.x + d, a.y + d, a.z + d}; }

Pos operator-(const Pos& a, double d) { return {a.x - d, a.y - d, a.z - d}; }

Pos operator*(const Pos& a, double d) { return {a.x * d, a.y * d, a.z * d}; }

BlockPos operator+(const BlockPos& a, const BlockPos& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

BlockPos operator-(const BlockPos& a, const BlockPos& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

BlockPos operator*(const BlockPos& a, int c) { return {a.x * c, a.y * c, a.z * c}; }
