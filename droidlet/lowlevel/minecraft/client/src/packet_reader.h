// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "encoder.h"
#include "event.h"
#include "types.h"

class PacketReader {
 public:
  PacketReader() {}
  PacketReader(int sock, EventHandler* eventHandler);
  std::thread startThread();

 private:
  int readPacket();
  int varintFromStream(long* v);
  void bufferExactly(uint8_t* buf, unsigned long n);
  uint8_t next();
  uint8_t peek();
  void skip(unsigned long n);
  void skipRest();
  std::vector<uint8_t> next(unsigned long n);
  uint64_t readBigEndian(int n);

  // Decode types
  long readVarint();
  std::string readString();
  uint8_t readByte() { return next(); }
  bool readBool() { return next() != 0; }
  uint16_t readUint16() { return readBigEndian(2); }
  int16_t readInt16() { return readBigEndian(2); }
  uint32_t readUint32() { return readBigEndian(4); }
  int32_t readInt32() { return readBigEndian(4); }
  uint64_t readUint64() { return readBigEndian(8); }
  int64_t readInt64() { return readBigEndian(8); }
  int readInt();
  float readFloat();
  double readDouble();
  float readAngle() { return (float)readByte() * 360 / 256; }
  std::string readUuid();
  BlockPos readPosition();
  Pos readDeltaPos();
  Block readBlock();
  std::string readChat();
  Slot readSlot();

  // Packets
  void setCompression();
  void loginSuccess();
  void keepAlive();
  void joinGame();
  void spawnPosition();
  void chunkData();
  void playerPositionAndLook();
  void blockChange();
  void multiBlockChange();
  void chatMessage();
  void playerListItem();
  void spawnPlayer();
  void entityMetadata();
  void entityRelativeMove();
  void entityLookAndRelativeMove();
  void entityTeleport();
  void openWindow();
  void windowItems();
  void setSlot();
  void entityLook();
  void entityHeadLook();
  void destroyEntities();
  void serverDifficulty();
  void spawnObject();
  void spawnMob();
  void updateHealth();
  void timeUpdate();
  void collectItem();
  void confirmTransaction();
  void entityEquipment();

  // Packet subcomponents (helpers)
  ChunkSectionBlocks chunkSectionBlocks();
  std::vector<std::pair<std::string, std::string>> playerListItemsAddPlayer(int n);

  // Fields
  int socket_;
  EventHandler* eventHandler_;
  std::vector<uint8_t> data_;
  std::set<int> ignoredPids_;
  size_t data_off_ = 0;
  int threshold_ = 0;
  bool inPlayState_ = false;  // false=login, true=play state
};

class ExitGracefully : std::runtime_error {
 public:
  ExitGracefully(char const* what) : runtime_error(what) {}
};
