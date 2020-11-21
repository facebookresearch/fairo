// Copyright (c) Facebook, Inc. and its affiliates.


#include <glog/logging.h>
#include <iomanip>

#include "../../client/src/big_endian.h"
#include "../../client/src/block_data.h"
#include "../../client/src/types.h"
#include "anvil_reader.h"
#include "cuberite_constants.h"
#include "hooks.h"
#include "logging_reader.h"
#include "stream_decoder.h"

using namespace std;

LoggingReader::LoggingReader(const string& loggingBinPath, vector<string> mcaPaths,
                             const string& name)
    : log_(loggingBinPath), name_(name) {
  for (string mcaPath : mcaPaths) {
    AnvilReader::readAnvilFile(gameState_.getBlockMap(), mcaPath);
  }

  // Read version
  uint16_t versionMajor = log_.readIntType<uint16_t>();
  uint16_t versionMinor = log_.readIntType<uint16_t>();

  LOG(INFO) << "Logger version " << versionMajor << ":" << versionMinor;
}

void LoggingReader::stepHook() {
  auto bufStart = log_.getCount();
  uint8_t hookId = log_.readIntType<uint8_t>();
  tick_ = log_.readIntType<uint64_t>();

  LOG(INFO) << "Process hook id " << (int)hookId << " @ " << bufStart;

  switch (hookId) {
    case hooks::PLAYER_SPAWNED: {
      auto eid = log_.readIntType<uint64_t>();
      auto name = log_.readString();
      auto pos = log_.readFloatPos();
      auto look = log_.readLook();

      if (name == name_) {
        gameState_.setName(name);
        gameState_.setEntityId(eid);
        gameState_.setPosition(pos);
        gameState_.setLook(look);
        eid_ = eid;
        isPlayerInGame_ = true;
      } else {
        // hack: use name as UUID
        gameState_.addPlayer(name, name);
        gameState_.setPlayer(name, eid, pos, look);
      }
      break;
    }

    case hooks::PLAYER_DESTROYED: {
      auto eid = log_.readIntType<uint64_t>();

      if (eid == eid_) {
        isPlayerInGame_ = false;
      } else {
        gameState_.removePlayer(eid);
      }
      break;
    }

    case hooks::PLAYER_MOVING: {
      auto eid = log_.readIntType<uint64_t>();
      log_.readFloatPos();  // old pos
      auto newPos = log_.readFloatPos();

      if (eid == eid_) {
        gameState_.setPosition(newPos);
      } else {
        gameState_.setPlayerPos(eid, newPos);
      }
      break;
    }

    case hooks::CHUNK_AVAILABLE: {
      log_.readIntType<int64_t>();  // cx
      log_.readIntType<int64_t>();  // cz
      break;
    }

    case hooks::BLOCK_SPREAD: {
      log_.readIntPos();
      log_.readIntType<int8_t>();
      // FIXME: do something
      break;
    }

    case hooks::CHAT: {
      log_.readIntType<uint64_t>();  // eid
      log_.readString();
      // FIXME: do something
      break;
    }

    case hooks::COLLECTING_PICKUP: {
      log_.readIntType<uint64_t>();  // eid
      log_.readItem();               // eid
      // FIXME: do something
      break;
    }

    case hooks::KILLED: {
      log_.readIntType<uint64_t>();  // eid
      // FIXME: do something
      break;
    }

    case hooks::PLAYER_BROKEN_BLOCK: {
      log_.readIntType<uint64_t>();  // eid
      auto pos = log_.readIntPos();
      log_.readIntType<uint8_t>();  // face
      log_.readBlock();             // block

      gameState_.getBlockMap().setBlock(pos, BLOCK_AIR);
      break;
    }

    case hooks::PLAYER_PLACED_BLOCK: {
      log_.readIntType<uint64_t>();  // eid
      auto pos = log_.readIntPos();
      auto block = log_.readBlock();

      gameState_.getBlockMap().setBlock(pos, block);
      break;
    }

    case hooks::PLAYER_USED_BLOCK: {
      log_.readIntType<uint64_t>();  // eid
      log_.readIntPos();
      log_.readIntType<int8_t>();
      log_.readFloat();
      log_.readFloat();
      log_.readFloat();
      log_.readBlock();
      // FIXME: do something
      break;
    }

    case hooks::PLAYER_USED_ITEM: {
      log_.readIntType<uint64_t>();  // eid
      // FIXME: do something
      break;
    }

    case hooks::SPAWNED_ENTITY: {
      log_.readIntType<uint64_t>();  // eid
      auto etype = log_.readIntType<uint8_t>();
      log_.readFloatPos();           // pos
      log_.readLook();               // look
      if (etype == cuberiteConstants::etMob) {
        log_.readIntType<uint8_t>();  // mtype
      }
      // FIXME: do something
      break;
    }

    case hooks::MONSTER_MOVED: {
      log_.readIntType<uint64_t>();  // eid
      log_.readFloatPos();  // pos
      log_.readLook();      // look
      // FIXME: do something
      break;
    }

    case hooks::PLAYER_LOOK: {
      auto eid = log_.readIntType<uint64_t>();
      auto look = log_.readLook();

      if (eid == eid_) {
        gameState_.setLook(look);
      } else {
        gameState_.setPlayerLook(eid, look);
      }
      break;
    }

    case hooks::TAKE_DAMAGE: {
      log_.readIntType<uint64_t>();  // eid
      auto dt = log_.readIntType<uint8_t>();
      log_.readDouble();             // final damage
      log_.readDouble();             // raw damage
      log_.readDouble();             // knockback x
      log_.readDouble();             // knockback y
      log_.readDouble();             // knockback z
      if (dt == cuberiteConstants::dtAttack) {
        log_.readIntType<uint64_t>();  // attacker eid
      }
      break;
    }

    case hooks::WORLD_STARTED: {
      log_.skip(40);  // hashes
      break;
    }

    default:
      LOG(FATAL) << "Can't handle hook id " << (int)hookId;
  }
}

void LoggingReader::stepToSpawn() {
  while (currentState().getName() != name_) {
    stepHook();
  }
}

void LoggingReader::stepTicks(unsigned long n) {
  auto targetTick = tick_ + n;
  unsigned long tick = -1;
  while (peekNextHookTick(&tick) && tick <= targetTick) {
    stepHook();
  }
  CHECK_LE(tick_, targetTick);
  tick_ = targetTick;
}

bool LoggingReader::peekNextHookTick(unsigned long* nextHookTick) {
  vector<uint8_t> bytes;
  bool valid = log_.peek(bytes, 9);
  if (!valid) return false;
  *nextHookTick = BigEndian::readIntType<uint64_t>(&bytes[1]);
  return true;
}

