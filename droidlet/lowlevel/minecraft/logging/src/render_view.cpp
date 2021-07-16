// Copyright (c) Facebook, Inc. and its affiliates.


#include <glog/logging.h>
#include <fstream>
#include <string>
#include <vector>

#include "../../client/src/graphics.h"
#include "../../client/src/types.h"
#include "../lib/argparse/argparse.hpp"
#include "anvil_reader.h"

using namespace std;

static int MAX_DEPTH = 2000;

int main(int argc, const char* argv[]) {
  google::InitGoogleLogging("render_views");

  ArgumentParser args;
  args.addArgument("--out-prefix", 1, false);
  args.addArgument("--mca-files", '+', false);
  args.addArgument("--camera", '+', false);
  args.addArgument("--sizes", '+', false);
  args.addArgument("--depth", 1, false);
  args.addArgument("--block", 1, false);
  args.addArgument("--blockpos", 1, false);
  args.addArgument("--look", '+', false);
  args.parse(argc, argv);

  string out_prefix = args.retrieve<string>("out-prefix");
  vector<string> mcaFiles = args.retrieve<vector<string>>("mca-files");
  vector<string> cameraArg = args.retrieve<vector<string>>("camera");
  vector<string> lookArg = args.retrieve<vector<string>>("look");
  vector<string> sizes = args.retrieve<vector<string>>("sizes");
  Pos camera = {stof(cameraArg[0]), stof(cameraArg[1]), stof(cameraArg[2])};
  Look look = {stof(lookArg[0]), stof(lookArg[1])};
  int width = stoi(sizes[0]);
  int height = stoi(sizes[1]);

  int block = stoi(args.retrieve<string>("block"));
  int depth = stoi(args.retrieve<string>("depth"));
  int blockpos = stoi(args.retrieve<string>("blockpos"));

  // Read block map into memory
  BlockMap blockMap;
  for (string mcaFile : mcaFiles) {
    AnvilReader::readAnvilFile(blockMap, mcaFile);
  }

  // Calculate vision
  vector<Block> blockVision;
  vector<float> depthVision;
  vector<BlockPos> blockPos;
  string fname;
  fname.reserve(256);

  LOG(INFO) << "Getting vision camera=" << camera << " yaw=" << look.yaw << " pitch=" << look.pitch;

  Graphics::vision(blockVision, depthVision, blockPos, height, width, blockMap, camera, look,
                   MAX_DEPTH, false/*moveToPlayerHead*/);

  if (depth) {
      // Write depth vision
      snprintf(&fname[0], fname.capacity(), "%s.depth.bin", out_prefix.c_str());
      ofstream df(fname, ios::binary);
      for (float ff : depthVision) {
          df.write((char*)&ff, sizeof(float));
      }
      df.close();
  }

  if (block) {
      // Write block vision
      snprintf(&fname[0], fname.capacity(), "%s.block.bin", out_prefix.c_str());
      ofstream bf(fname, ios::binary);
      for (Block b : blockVision) {
          bf.write((char*)&b, sizeof(Block));
      }
      bf.close();
  }

  if (blockpos) {
      // Write block pos
      snprintf(&fname[0], fname.capacity(), "%s.blockpos.bin", out_prefix.c_str());
      ofstream bpf(fname, ios::binary);
      for (auto bp : blockPos) {
          bpf.write((char*)&bp, sizeof(BlockPos));
      }
      bpf.close();
  }

  return 0;
}
