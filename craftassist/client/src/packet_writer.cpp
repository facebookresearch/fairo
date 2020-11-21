// Copyright (c) Facebook, Inc. and its affiliates.

#include <sys/socket.h>
#include <iostream>
#include <vector>

#include "packet_writer.h"

using namespace std;

void PacketWriter::safeWrite(vector<uint8_t> bs) {
  lock_.lock();
  socketSend(bs);
  lock_.unlock();
}

void PacketWriter::socketSend(vector<uint8_t> bs) { send(socket_, &bs[0], bs.size(), 0); }
