// Copyright (c) Facebook, Inc. and its affiliates.


#include <glog/logging.h>
#include <zlib.h>
#include <fstream>
#include <vector>

#include "../../client/src/big_endian.h"
#include "../../client/src/nbt_tag.h"
#include "anvil_reader.h"

#define BUF_SIZE (16 * 1024)
#define UBUF_SIZE (512 * 1024)

#define SECTION_SIZE 4096
#define BLOCKS_PER_CHUNK 4096  // 16x16x16 blocks

using namespace std;

void AnvilReader::readAnvilFile(BlockMap& m, const string& path) {
  uint8_t buf[BUF_SIZE];
  uint8_t ubuf[UBUF_SIZE];

  ifstream is(path, ifstream::binary);
  if (!is) {
    LOG(FATAL) << "Failed to open " << path;
  }

  // Read chunk locations
  vector<long> chunkOffsets;
  is.read((char*)buf, SECTION_SIZE);
  int k = 0;
  for (int z = 0; z < 32; z++) {
    for (int x = 0; x < 32; x++, k += 4) {
      int offset = BigEndian::readBigEndian(buf + k, 3);
      int sectorCount = buf[k + 3];

      if (offset == 0 && sectorCount == 0) {
        continue;
      }

      chunkOffsets.push_back(offset);
    }
  }
  LOG(INFO) << "Found " << chunkOffsets.size() << " chunks";

  // Read each chunk
  for (long offset : chunkOffsets) {
    // Read length and compression type
    is.seekg(offset * SECTION_SIZE);
    is.read((char*)buf, 5);
    auto length = BigEndian::readIntType<uint32_t>(buf);
    CHECK_EQ(buf[4], 2) << "Bad compression type: " << (int)buf[4];

    // Read compressed bytes
    CHECK((length - 1) < BUF_SIZE) << (length - 1) << " too big for buf";
    is.read((char*)buf, length - 1);
    uLongf destLen = UBUF_SIZE;
    int err = uncompress(ubuf, &destLen, buf, length - 1);
    CHECK_EQ(err, Z_OK) << "zlib uncompress error: " << err;

    // Read NBT data
    uint8_t* p = ubuf;
    shared_ptr<NBTTagCompound> root = static_pointer_cast<NBTTagCompound>(NBTTag::from(&p));
    auto level = static_pointer_cast<NBTTagCompound>(root->getChild("Level"));
    int cx = static_pointer_cast<NBTTagInt>(level->getChild("xPos"))->getVal();
    int cz = static_pointer_cast<NBTTagInt>(level->getChild("zPos"))->getVal();
    auto sections = static_pointer_cast<NBTTagList>(level->getChild("Sections"));

    for (auto section : sections->getVals()) {
      // Set chunk sections with blocks
      auto s = static_pointer_cast<NBTTagCompound>(section);
      int cy = static_pointer_cast<NBTTagByte>(s->getChild("Y"))->getVal();
      CHECK_EQ(s->getChild("Add"), shared_ptr<NBTTag>(NULL)) << "Not implemented: NBT Section Add";
      auto ids = static_pointer_cast<NBTTagByteArray>(s->getChild("Blocks"))->getVal();
      auto metas = static_pointer_cast<NBTTagByteArray>(s->getChild("Data"))->getVal();

      LOG(INFO) << "Setting chunk x=" << cx << " y=" << cy << " z=" << cz;
      m.setChunk(getChunkSection(ids, metas, cx, cy, cz));
    }

    for (int cy = 0; cy < 16; cy++) {
      // Set all-air chunks that were not included above
      if (!m.chunkExists(cx, cy, cz)) {
        LOG(INFO) << "Setting all-air chunk x=" << cx << " y=" << cy << " z=" << cz;
        m.setChunk({cx, cy, cz, NULL});
      }
    }
  }
}

ChunkSection AnvilReader::getChunkSection(const vector<uint8_t>& ids, const vector<uint8_t>& metas,
                                          int cx, int cy, int cz) {
  CHECK_EQ(ids.size(), BLOCKS_PER_CHUNK);        // 1 byte per id
  CHECK_EQ(metas.size(), BLOCKS_PER_CHUNK / 2);  // 1 nibble per meta
  ChunkSectionBlocks blocks = make_shared<array<Block, BLOCKS_PER_CHUNK>>();
  for (int i = 0; i < BLOCKS_PER_CHUNK; i++) {
    (*blocks)[i].id = ids[i];
    (*blocks)[i].meta = ((i & 1) == 0) ? (metas[i / 2] >> 4) : (metas[i / 2] & 0xf);
  }
  return ChunkSection{cx, cy, cz, blocks};
}
