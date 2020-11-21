// Copyright (c) Facebook, Inc. and its affiliates.


#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "../../client/src/nbt_tag.h"

using namespace std;

void toNumpy(uint8_t* from, ostream& out) {
  auto schematic = static_pointer_cast<NBTTagCompound>(NBTTag::from(&from));

  // get schematic size
  short xSize = static_pointer_cast<NBTTagShort>(schematic->getChild("Width"))->getVal();
  short ySize = static_pointer_cast<NBTTagShort>(schematic->getChild("Height"))->getVal();
  short zSize = static_pointer_cast<NBTTagShort>(schematic->getChild("Length"))->getVal();

  // get schematic data
  vector<uint8_t> ids =
      static_pointer_cast<NBTTagByteArray>(schematic->getChild("Blocks"))->getVal();
  vector<uint8_t> metas =
      static_pointer_cast<NBTTagByteArray>(schematic->getChild("Data"))->getVal();
  CHECK_EQ(ids.size(), xSize * ySize * zSize);
  CHECK_EQ(metas.size(), xSize * ySize * zSize);

  // .npy header
  string magic = "\x93NUMPY";
  char version[] = "\x01\x00";
  ostringstream heads;
  heads << "{\"descr\": \"uint8\", \"fortran_order\": False, \"shape\": (" << ySize << "," << zSize
        << "," << xSize << ", 2)}";
  int size = magic.size() + 2 + heads.str().size() + 1;  // 2 = HEADER_LEN, 1 = \n
  int padding = 16 - (size % 16);
  for (int i = 0; i < padding - 1; i++) {
    heads << ' ';
  }
  heads << '\n';
  string head = heads.str();

  // write npy header
  out << magic;
  out.write(version, 2);
  uint16_t headSize = head.size();
  out.write((char*)&headSize, 2);
  out << head;

  // write data
  for (unsigned long i = 0; i < ids.size(); i++) {
    out << ids[i] << metas[i];
  }
}

int main(int argc, const char** argv) {
  google::InitGoogleLogging("schematic_convert");

  CHECK_EQ(argc, 3) << "Usage: " << argv[0] << " schematic_file out_file";

  ifstream is(argv[1], ios::binary | ios::ate);
  CHECK(is);
  long size = is.tellg();
  is.seekg(0, ios::beg);

  // read whole file into buf
  uint8_t buf[size];
  is.read((char*)buf, size);
  is.close();

  // write .npy file
  ofstream os(argv[2], ios::binary);
  toNumpy(buf, os);
  os.close();

  return 0;
}
