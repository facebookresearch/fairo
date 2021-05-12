// Copyright (c) Facebook, Inc. and its affiliates.


#pragma once
#include <string>

#include "../../client/src/block_map.h"
#include "../../client/src/types.h"

// This class provides functions to read a .mca region file, outlined here:
//
// https://minecraft.gamepedia.com/Anvil_file_format
// https://minecraft.gamepedia.com/Region_file_format
//
class AnvilReader {
 public:
  // Read a region file for region, and set the BlockMap chunks
  //
  // N.B. this ONLY reads blocks. No entities or other data is read.
  static void readAnvilFile(BlockMap& m, const std::string& path);

 private:
  static ChunkSection getChunkSection(const std::vector<uint8_t>& ids,
                                      const std::vector<uint8_t>& metas, int cx, int cy, int cz);
};
