// Copyright (c) Facebook, Inc. and its affiliates.


#include <glog/logging.h>
#include <string>
#include <vector>

#include "../../client/src/graphics.h"
#include "../../client/src/types.h"
#include "../lib/argparse/argparse.hpp"
#include "logging_reader.h"

using namespace std;

int main(int argc, const char* argv[]) {
  ArgumentParser args;
  args.addArgument("--out-dir", 1, false);
  args.addArgument("--log-file", 1, false);
  args.addArgument("--name", 1, false);
  args.addArgument("--mca-files", '+', false);
  args.parse(argc, argv);

  string outdir = args.retrieve<string>("out-dir");
  string logfile = args.retrieve<string>("log-file");
  string playerName = args.retrieve<string>("name");
  vector<string> mcaFiles = args.retrieve<vector<string>>("mca-files");

  string fname;
  fname.reserve(256);

  int height = 128;
  int width = 128;
  int maxDepth = 48;
  vector<Block> blockVision;
  vector<float> depthVision;
  vector<BlockPos> blockPos;

  LoggingReader loggingReader(logfile, mcaFiles, playerName);
  loggingReader.stepToSpawn();

  for (int i = 0; loggingReader.isPlayerInGame(); i++) {
    Graphics::vision(blockVision, depthVision, blockPos, height, width,
                     loggingReader.currentState().getBlockMap(),
                     loggingReader.currentState().getPosition(),
                     loggingReader.currentState().getLook(), maxDepth);

    snprintf(&fname[0], fname.capacity(), "%s/depth.%08u.bin", outdir.c_str(), i);
    ofstream df(fname, ios::binary);
    for (float ff : depthVision) {
      df.write((char*)&ff, sizeof(float));
    }
    df.close();

    snprintf(&fname[0], fname.capacity(), "%s/block.%08u.bin", outdir.c_str(), i);
    ofstream bf(fname, ios::binary);
    for (Block b : blockVision) {
      bf.write((char*)&b, 1);
    }
    bf.close();

    loggingReader.stepTicks(1);
  }

  return 0;
}
