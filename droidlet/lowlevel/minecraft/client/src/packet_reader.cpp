// Copyright (c) Facebook, Inc. and its affiliates.

#include <glog/logging.h>
#include <sys/socket.h>
#include <zlib.h>
#include <array>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <iostream>
#include <memory>
#include <thread>

#include "../lib/rapidjson/include/rapidjson/document.h"
#include "../lib/rapidjson/include/rapidjson/stringbuffer.h"
#include "event.h"
#include "event_handler.h"
#include "nbt_tag.h"
#include "packet_reader.h"
#include "types.h"
#include "util.h"

using namespace std;

PacketReader::PacketReader(int sock, EventHandler* eventHandler) {
  socket_ = sock;
  eventHandler_ = eventHandler;
}

thread PacketReader::startThread() {
  thread t([&]() {
    try {
      while (true) {
        int pid = readPacket();
        if (!inPlayState_) {
          // login state
          switch (pid) {
            case 0x02:
              loginSuccess();
              break;
            case 0x03:
              setCompression();
              break;
            default:
              LOG(FATAL) << "Bad pid 0x" << hex << pid << " in login state\n";
          }
        } else {
          // play state: most stuff happens here
          switch (pid) {
            case 0x00:
              spawnObject();
              break;
            case 0x03:
              spawnMob();
              break;
            case 0x05:
              spawnPlayer();
              break;
            case 0x0b:
              blockChange();
              break;
            case 0x0d:
              serverDifficulty();
              break;
            case 0x0f:
              chatMessage();
              break;
            case 0x10:
              multiBlockChange();
              break;
            case 0x11:
              confirmTransaction();
              break;
            case 0x13:
              openWindow();
              break;
            case 0x14:
              windowItems();
              break;
            case 0x16:
              setSlot();
              break;
            case 0x1f:
              keepAlive();
              break;
            case 0x20:
              chunkData();
              break;
            case 0x23:
              joinGame();
              break;
            case 0x26:
              entityRelativeMove();
              break;
            case 0x27:
              entityLookAndRelativeMove();
              break;
            case 0x28:
              entityLook();
              break;
            case 0x2e:
              playerListItem();
              break;
            case 0x2f:
              playerPositionAndLook();
              break;
            case 0x32:
              destroyEntities();
              break;
            case 0x36:
              entityHeadLook();
              break;
            case 0x3c:
              entityMetadata();
              break;
            case 0x3f:
              entityEquipment();
              break;
            case 0x4c:
              entityTeleport();
              break;
            case 0x46:
              spawnPosition();
              break;
            case 0x41:
              updateHealth();
              break;
            case 0x47:
              timeUpdate();
              break;
            case 0x4b:
              collectItem();
              break;
            default:
              if (ignoredPids_.count(pid) == 0) {
                LOG(INFO) << "ignored pid=0x" << hex << pid;
                ignoredPids_.insert(pid);
              }
              data_.clear();
              data_off_ = 0;
          }
        }
      }
    } catch (ExitGracefully* e) {
    }
  });
  return t;
}

////////////////
// Private

// Read entire next packet into data_ and return the pid
int PacketReader::readPacket() {
  if (data_.size() != data_off_) {
    LOG(FATAL) << "unread data, size=" << data_.size() << " off=" << data_off_;
  }
  data_off_ = 0;

  // Read packet packet length
  long packetLen;
  varintFromStream(&packetLen);
  unsigned bytesLeft = packetLen;

  // If compression enabled, read dataLen: length of uncompressed pid+data, or 0
  long dataLen = 0;
  if (threshold_ > 0) {
    bytesLeft -= varintFromStream(&dataLen);
  }

  // Read pid + data into data_
  if (dataLen != 0) {
    // zlib-encoded: read first into zlib_buf
    uint8_t zlib_buf[bytesLeft];
    bufferExactly(zlib_buf, bytesLeft);
    data_.resize(dataLen);
    unsigned long destLen = dataLen;
    int err = uncompress(&data_[0], &destLen, zlib_buf, bytesLeft);
    if (err != 0) {
      LOG(WARNING) << "Closing PacketReader since uncompress returned " << err;
      throw new ExitGracefully("uncompress failed");
    }
    CHECK_EQ(destLen, dataLen);
  } else {
    // Not zlib-encoded: read directly into data_
    data_.resize(bytesLeft);
    bufferExactly(&data_[0], bytesLeft);
  }

  // extract pid and return
  return readVarint();
}

void PacketReader::bufferExactly(uint8_t* buf, unsigned long n) {
  unsigned int off = 0;
  while (off < n) {
    int r = recv(socket_, (buf + off), (n - off), 0);
    if (r <= 0) {
      throw new ExitGracefully("");
    }
    off += r;
  }
}

// Read a varint from socket_ into v, and return the number of bytes read
int PacketReader::varintFromStream(long* v) {
  long read = 0;
  *v = 0;
  uint8_t b;
  while (true) {
    bufferExactly(&b, 1);
    long val = b & 127;
    *v |= (val << (7 * read));
    read += 1;
    if ((b & 128) == 0) {
      return read;
    }
  }
}

// Get next byte in packet
uint8_t PacketReader::next() { return data_[data_off_++]; }

// Peek at next byte in packet without moving the cursor
uint8_t PacketReader::peek() { return data_[data_off_]; }

// Skip n bytes in packet
void PacketReader::skip(unsigned long n) { data_off_ += n; }

// Skip remaining bytes in packet
void PacketReader::skipRest() { data_off_ = data_.size(); }

uint64_t PacketReader::readBigEndian(int n) {
  uint64_t out = 0;
  for (int i = 0; i < n; i++) {
    out <<= 8;
    out |= next();
  }
  return out;
}

////////////////
// Decode Types

int PacketReader::readInt() {
  uint32_t bytes = readBigEndian(4);
  int f;
  memcpy(&f, &bytes, 4);
  return f;
}

long PacketReader::readVarint() {
  long read = 0;
  long out = 0;
  while (true) {
    uint8_t b = next();
    long val = b & 127;
    out |= (val << (7 * read));
    read += 1;
    if ((b & 128) == 0) {
      return out;
    }
  }
}

string PacketReader::readString() {
  auto length = readVarint();
  string s(data_.begin() + data_off_, data_.begin() + data_off_ + length);
  data_off_ += length;
  return s;
}

float PacketReader::readFloat() {
  uint32_t bytes = readBigEndian(4);
  float f;
  memcpy(&f, &bytes, 4);
  return f;
}

double PacketReader::readDouble() {
  uint64_t bytes = readBigEndian(8);
  double d;
  memcpy(&d, &bytes, 8);
  return d;
}

string PacketReader::readUuid() {
  unsigned char data[16];
  for (int i = 0; i < 16; i++) {
    data[i] = next();
  }
  boost::uuids::uuid uuid;
  memcpy(&uuid, data, 16);
  return boost::uuids::to_string(uuid);
}

BlockPos PacketReader::readPosition() {
  long v = readUint64();
  int x = v >> 38;
  int y = (v >> 26) & 0xfff;
  int z = v << 38 >> 38;
  if (x >= (1 << 25)) {
    x -= (1 << 26);
  }
  if (y >= (1 << 11)) {
    y -= (1 << 12);
  }
  if (z >= (1 << 25)) {
    z -= (1 << 26);
  }
  return BlockPos{x, y, z};
}

// See http://wiki.vg/Protocol#Entity_Look_And_Relative_Move
Pos PacketReader::readDeltaPos() {
  short dx = readInt16();
  short dy = readInt16();
  short dz = readInt16();
  return {
      (double)dx / 4096, (double)dy / 4096, (double)dz / 4096,
  };
}

Block PacketReader::readBlock() {
  int v = readVarint();
  Block block;
  block.id = (v >> 4);
  block.meta = (v & 0xf);
  return block;
}

string PacketReader::readChat() {
  string s = readString();
  rapidjson::Document j;
  j.Parse(s.c_str());

  // Pull out text fields, ignore formatting.
  string text = j["text"].GetString();
  for (auto e = j["extra"].Begin(); e != j["extra"].End(); e++) {
    text += (*e)["text"].GetString();
  }
  return text;
}

Slot PacketReader::readSlot() {
  int16_t blockId = readInt16();
  if (blockId == -1) {
    return EMPTY_SLOT;
  }
  uint8_t count = readByte();
  uint16_t damage = readUint16();
  if (peek() != 0) {
    uint8_t* p = &data_[data_off_];
    shared_ptr<NBTTag> tag = NBTTag::from(&p);  // read nbt data
    skip(p - &data_[data_off_]);                // skip read nbt data
  } else {
    next();  // skip 0 byte indicating no NBT data
  }
  // N.B. Cuberite seems to send block meta in the "damage" field
  uint8_t meta = damage;
  return {(uint16_t)blockId, meta, count, 0};
}

////////////////
// Packets

void PacketReader::setCompression() {
  threshold_ = readVarint();
  LOG(INFO) << "Set threshold to " << threshold_;
}

void PacketReader::loginSuccess() {
  string uuid = readString();
  string username = readString();
  inPlayState_ = true;

  LoginSuccessEvent e = {uuid, username};
  eventHandler_->handle(e);
}

void PacketReader::keepAlive() {
  auto keepaliveId = readUint64();

  KeepaliveEvent e = {keepaliveId};
  eventHandler_->handle(e);
}

void PacketReader::chunkData() {
  auto cx = readInt32();
  auto cz = readInt32();
  readByte();  // skip groundUpContinuous
  auto bitmask = readVarint();
  readVarint();  // skip data_size

  array<ChunkSection, 16> chunks;
  for (int cy = 0; cy < 16; cy++) {
    auto blocks = (bitmask & 1) ? chunkSectionBlocks() : NULL;
    chunks[cy] = {cx, cy, cz, blocks};
    bitmask >>= 1;
  }
  skipRest();

  ChunkDataEvent e = {cx, cz, chunks};
  eventHandler_->handle(e);
}

void PacketReader::joinGame() {
  auto entityId = readUint32();
  uint8_t gameModeByte = readByte();
  readUint32();  // dimension
  readByte();    // difficulty
  readByte();    // maxPlayers
  readString();  // levelType
  readByte();    // reducedDebugInfo

  GameMode gameMode;
  if (gameModeByte == 0) {
    gameMode = GameMode::SURVIVAL;
  } else if (gameModeByte == 1) {
    gameMode = GameMode::CREATIVE;
  } else {
    LOG(FATAL) << "Can't handle game mode: " << (int)gameModeByte;
  }

  JoinGameEvent e = {entityId, gameMode};
  eventHandler_->handle(e);
}

void PacketReader::entityRelativeMove() {
  unsigned long entityId = readVarint();
  Pos deltaPos = readDeltaPos();
  skipRest();  // on_ground

  EntityRelativeMoveEvent e = {entityId, deltaPos};
  eventHandler_->handle(e);
}

void PacketReader::entityLookAndRelativeMove() {
  unsigned long entityId = readVarint();
  Pos deltaPos = readDeltaPos();
  float yaw = readAngle();
  float pitch = readAngle();
  skipRest();  // on_ground

  Look look = {yaw, pitch};
  EntityLookAndRelativeMoveEvent e = {entityId, deltaPos, look};
  eventHandler_->handle(e);
}

void PacketReader::entityLook() {
  unsigned long entityId = readVarint();
  float yaw = readAngle();
  float pitch = readAngle();
  skipRest();  // on_ground

  Pos deltaPos = {0, 0, 0};
  Look look = {yaw, pitch};
  EntityLookAndRelativeMoveEvent e = {entityId, deltaPos, look};
  eventHandler_->handle(e);
}

void PacketReader::playerListItem() {
  auto action = readVarint();
  auto numPlayers = readVarint();
  if (action == 0) {
    // add players
    vector<pair<string, string>> uuidNamePairs = playerListItemsAddPlayer(numPlayers);
    AddPlayersEvent e = {uuidNamePairs};
    eventHandler_->handle(e);
  } else if (action == 4) {
    // remove players
    vector<string> uuidsToRemove;
    for (int i = 0; i < numPlayers; i++) {
      uuidsToRemove.push_back(readUuid());
    }
    RemovePlayersEvent e = {uuidsToRemove};
    eventHandler_->handle(e);
  } else if (action < 4) {
    // Not implemented:
    // 1: update gamemode
    // 2: update latency
    // 3: update display name
    skipRest();
  } else {
    LOG(FATAL) << "Bad PlayerListItem action: " << action;
  }
}

void PacketReader::playerPositionAndLook() {
  Pos pos = {readDouble(), readDouble(), readDouble()};
  Look look = {readFloat(), readFloat()};
  auto flags = readByte();
  auto teleportId = readVarint();

  PlayerPositionAndLookEvent e = {pos, look, flags, teleportId};
  eventHandler_->handle(e);
}

void PacketReader::entityHeadLook() {
  unsigned long entityId = readVarint();
  float yaw = readAngle();

  EntityHeadLookEvent e = {entityId, yaw};
  eventHandler_->handle(e);
}

void PacketReader::destroyEntities() {
  unsigned long count = readVarint();
  vector<uint64_t> entityIds;
  for (int i = 0; i < (long)count; i++) {
    entityIds.push_back(readVarint());
  }
  DestroyEntitiesEvent e = {count, entityIds};
  eventHandler_->handle(e);
}

// TODO: fix hacky
void PacketReader::entityMetadata() {
  unsigned long entityId = readVarint();
  uint8_t index = readByte();
  while (index != 0xff) {
    readVarint();
    switch (index) {
      case 0: {
        readByte();
        break;
      }
      case 1: {
        readVarint();
        break;
      }
      case 2: {
        readFloat();
        break;
      }
      case 3: {
        readString();
        break;
      }
      case 4: {
        readChat();
        break;
      }
      case 5: {
        bool val5 = readBool();
        if (val5) {
          readChat();
        }
        break;
      }
      case 6: {
        Slot item = readSlot();
        SpawnItemStackEvent e = {entityId, item};
        eventHandler_->handle(e);
        break;
      }
      default: {
        skipRest();
        return;
      }
    }
    index = readByte();
  }
}

void PacketReader::entityTeleport() {
  unsigned long entityId = readVarint();
  double x = readDouble();
  double y = readDouble();
  double z = readDouble();
  float yaw = readAngle();
  float pitch = readAngle();
  skipRest();  // on_ground

  Pos pos = {x, y, z};
  Look look = {yaw, pitch};
  EntityTeleportEvent e = {entityId, pos, look};
  eventHandler_->handle(e);
}

void PacketReader::spawnPosition() {
  auto pos = readPosition();

  SpawnPositionEvent e = {pos};
  eventHandler_->handle(e);
}

void PacketReader::spawnPlayer() {
  unsigned long entityId = readVarint();
  string uuid = readUuid();
  double x = readDouble();
  double y = readDouble();
  double z = readDouble();
  float yaw = readAngle();
  float pitch = readAngle();
  skipRest();  // entity metadata

  Pos pos = {x, y, z};
  Look look = {yaw, pitch};
  SpawnPlayerEvent e = {entityId, uuid, pos, look};
  eventHandler_->handle(e);
}

void PacketReader::blockChange() {
  BlockPos pos = readPosition();
  Block block = readBlock();

  BlockChangeEvent e = {pos, block};
  eventHandler_->handle(e);
}

void PacketReader::serverDifficulty() {
  uint8_t difficulty = readByte();

  ServerDifficultyEvent e = {difficulty};
  eventHandler_->handle(e);
}

void PacketReader::chatMessage() {
  string chat = readChat();
  uint8_t position = readByte();

  if (position != 0) {
    // Only handle player-initiated chat
    return;
  }

  ChatMessageEvent e = {chat, position};
  eventHandler_->handle(e);
}

void PacketReader::multiBlockChange() {
  int cx = readInt32();
  int cz = readInt32();
  int count = readVarint();
  for (int i = 0; i < count; i += 1) {
    uint8_t pos = readByte();
    uint8_t y = readByte();
    Block block = readBlock();
    auto ox = (pos >> 4);   // upper 4b
    auto oz = (pos & 0xf);  // lower 4b
    BlockPos target = {(cx << 4) | ox, y, (cz << 4) | oz};

    BlockChangeEvent e = {target, block};
    eventHandler_->handle(e);
  }
}

void PacketReader::confirmTransaction() {
  uint8_t windowId = readByte();
  uint16_t counter = readUint16();
  bool accepted = readBool();

  ConfirmTransactionEvent e = {windowId, counter, accepted};
  eventHandler_->handle(e);
}

void PacketReader::openWindow() {
  uint8_t windowId = readByte();
  string windowTypeString = readString();
  skipRest();  // window title, # slots, entity id

  WindowType windowType;
  if (windowTypeString == "minecraft:crafting_table") {
    windowType = WindowType::CRAFTING_TABLE;
  } else {
    LOG(FATAL) << "Can't handle openWindow type=" << windowTypeString;
  }

  OpenWindowEvent e = {windowId, windowType};
  eventHandler_->handle(e);
}

void PacketReader::windowItems() {
  uint8_t windowId = readByte();
  uint16_t count = readUint16();

  vector<Slot> slots;
  for (auto i = 0; i < count; i++) {
    slots.push_back(readSlot());
  }

  WindowItemsEvent e = {windowId, slots};
  eventHandler_->handle(e);
}

void PacketReader::setSlot() {
  uint8_t windowId = readByte();
  uint16_t index = readInt16();
  Slot slot = readSlot();

  SetSlotEvent e = {windowId, index, slot};
  eventHandler_->handle(e);
}

void PacketReader::spawnObject() {
  uint64_t entityId = readVarint();
  string uuid = readUuid();
  uint8_t objectType = readByte();
  double x = readDouble();
  double y = readDouble();
  double z = readDouble();

  skipRest();  // look, head angle, velocity

  if (objectType == 2) {  // rn only put item stacks into object map
    Pos pos = {x, y, z};
    SpawnObjectEvent e = {entityId, uuid, objectType, pos};
    eventHandler_->handle(e);
  }
}

void PacketReader::spawnMob() {
  uint64_t entityId = readVarint();
  string uuid = readUuid();
  uint8_t mobType = readVarint();
  double x = readDouble();
  double y = readDouble();
  double z = readDouble();
  float yaw = readAngle();
  float pitch = readAngle();

  skipRest();  // head angle, velocity, metadata

  Pos pos = {x, y, z};
  Look look = {yaw, pitch};
  SpawnMobEvent e = {entityId, uuid, mobType, pos, look};
  eventHandler_->handle(e);
}

void PacketReader::updateHealth() {
  float health = readFloat();
  uint32_t foodLevel = readVarint();
  skipRest();  // skip food saturation level from the packet

  UpdateHealthEvent e = {health, foodLevel};
  eventHandler_->handle(e);
}

void PacketReader::timeUpdate() {
  long worldAge = readInt64();
  long timeOfDay = readInt64();
  TimeUpdateEvent e = {worldAge, timeOfDay};
  eventHandler_->handle(e);
}

void PacketReader::collectItem() {
  uint32_t collectedEntityId = readVarint();
  uint32_t collectorEntityId = readVarint();
  uint8_t pickedItemCount = readVarint();
  skipRest();
  CollectItemEvent e = {collectedEntityId, collectorEntityId, pickedItemCount};
  eventHandler_->handle(e);
}

void PacketReader::entityEquipment() {
  uint64_t entityId = readVarint();
  uint8_t which = readVarint();
  Slot slot = readSlot();

  EntityEquipmentEvent e = {entityId, which, slot};
  eventHandler_->handle(e);
}

////////////////
// Packet subcomponents (helpers)
//
// See http://wiki.vg/Chunk_Format#Chunk_Section_structure
ChunkSectionBlocks PacketReader::chunkSectionBlocks() {
  auto bitsPerBlock = readByte();
  CHECK_EQ(bitsPerBlock, 13);
  uint64_t blockIdMask = (((1 << bitsPerBlock) - 1) ^ 0xf);  // bits 4-12

  // Skip palette
  int paletteLen = readVarint();
  for (int i = 0; i < paletteLen; i++) {
    readVarint();
  }

  // Read block data
  auto dataLen = readVarint();  // number of longs to read
  const int DATA_LEN = 4096 * 13 / 64;
  CHECK_EQ(dataLen, DATA_LEN);
  array<uint64_t, DATA_LEN> longs;
  for (int i = 0; i < DATA_LEN; i++) {
    longs[i] = readUint64();
  }
  ChunkSectionBlocks blocks = make_shared<array<Block, 4096>>();
  long bo = 0;
  for (long i = 0; i < 4096; i++) {
    // Because of the bit offset, block_ids might stretch across two longs
    auto startLong = bo / 64;
    auto off = bo % 64;
    auto endLong = (bo + bitsPerBlock - 1) / 64;
    uint64_t d = (startLong == endLong)
                     ? longs[startLong] >> off
                     : (longs[endLong] << (64 - off)) + (longs[startLong] >> off);
    auto blockId = (d & blockIdMask) >> 4;
    uint8_t blockMeta = d & 0xf;
    CHECK(blockId < 256) << "invalid block id > 256, " << blockId;
    (*blocks)[i] = Block{uint8_t(blockId), blockMeta};
    bo += bitsPerBlock;
  }
  skip(4096 / 2);  // block_light (nibble per block)
  skip(4096 / 2);  // sky_light (nibble per block)
  return blocks;
}

vector<pair<string, string>> PacketReader::playerListItemsAddPlayer(int numPlayers) {
  vector<pair<string, string>> pairs;
  for (int i = 0; i < numPlayers; i++) {
    string uuid = readUuid();
    string name = readString();
    auto numProperties = readVarint();
    for (int j = 0; j < numProperties; j++) {
      readString();      // property name
      readString();      // property value
      if (readBool()) {  // is signed?
        readString();    // signature
      }
    }
    readVarint();  // game mode
    readVarint();  // ping (ms)
    string displayName = readBool() ? readString() : "";

    pairs.push_back(make_pair(uuid, name));
  }
  return pairs;
}
