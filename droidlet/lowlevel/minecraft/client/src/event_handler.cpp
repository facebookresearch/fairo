// Copyright (c) Facebook, Inc. and its affiliates.

#include "event_handler.h"
#include <glog/logging.h>
#include "encoder.h"
#include "event.h"
#include "game_state.h"
#include "packet_writer.h"
#include "types.h"

using namespace std;
using std::optional;
using std::nullopt;

// Consider the client loaded when some number of chunks have been received
static const int CHUNK_DATA_EVENTS_LOADED_THRESHOLD = 50;

EventHandler::EventHandler(GameState* gameState, PacketWriter* packetWriter, Encoder* encoder) {
  gameState_ = gameState;
  packetWriter_ = packetWriter;
  encoder_ = encoder;
}

void EventHandler::handle(KeepaliveEvent e) {
  packetWriter_->safeWrite(encoder_->keepalivePacket(e.id));
}

void EventHandler::handle(LoginSuccessEvent e) {
  gameState_->setName(e.username);
  gameState_->setUuid(e.uuid);
  LOG(INFO) << "Login success! name=" << e.username << " uuid=" << e.uuid;
}

void EventHandler::handle(JoinGameEvent e) {
  if (e.gameMode > 1) {
    LOG(FATAL) << "Can't handle game mode " << e.gameMode;
  }
  gameState_->setEntityId(e.entityId);
  gameState_->setGameMode(e.gameMode);
  LOG(INFO) << "Entity id " << e.entityId;
  LOG(INFO) << "Creative mode: " << (bool)e.gameMode;
}

void EventHandler::handle(SpawnPositionEvent e) {
  gameState_->setPosition(e.pos);
  LOG(INFO) << "Spawn position: " << e.pos;
}

void EventHandler::handle(ChunkDataEvent e) {
  for (ChunkSection chunk : e.chunks) {
    gameState_->getBlockMap().setChunk(chunk);
  }
  if (++chunkDataEvents_ == CHUNK_DATA_EVENTS_LOADED_THRESHOLD) {
    LOG(INFO) << "Loaded! Unlocking main client thread";
    triggerCondition();
  }
}

void EventHandler::handle(PlayerPositionAndLookEvent e) {
  gameState_->setPosition(e.pos);
  gameState_->setLook(e.look);
  if (e.flags != 0) {
    LOG(FATAL) << "Can't handle flags: " << (int)e.flags;
  }
  packetWriter_->safeWrite(encoder_->teleportConfirmPacket(e.teleportId));
}

void EventHandler::handle(BlockChangeEvent e) {
  gameState_->getBlockMap().setBlock(e.pos, e.block);
  gameState_->getChangedBlocks().push({e.pos, e.block});
  if (blockChangeCondition_ && *blockChangeCondition_ == e.pos) {
    triggerCondition();
  }
}

void EventHandler::handle(ChatMessageEvent e) { gameState_->addChat(e.chat); }

void EventHandler::handle(AddPlayersEvent e) {
  for (pair<string, string> uuidNamePair : e.uuidNamePairs) {
    if (uuidNamePair.first.compare(gameState_->getUuid()) == 0) {
      // Don't store my own Player
      continue;
    }
    LOG(INFO) << "Add player uuid=" << uuidNamePair.first << " name=" << uuidNamePair.second;
    gameState_->addPlayer(uuidNamePair.first, uuidNamePair.second);
  }
}

void EventHandler::handle(SpawnPlayerEvent e) {
  LOG(INFO) << "Spawn player uuid=" << e.uuid << " eid=" << e.entityId << " pos=" << e.pos;
  gameState_->setPlayer(e.uuid, e.entityId, e.pos, e.look);
}

void EventHandler::handle(RemovePlayersEvent e) {
  for (string uuid : e.uuidsToRemove) {
    LOG(INFO) << "Remove player uuid=" << uuid;
    gameState_->removePlayer(uuid);
  }
}

void EventHandler::handle(EntityRelativeMoveEvent e) {
  optional<Mob> mob = gameState_->getMob(e.entityId);
  if (mob) {
    mob->pos = mob->pos + e.deltaPos;
    gameState_->setMob(*mob);
  } else {
    gameState_->setPlayerDeltaPos(e.entityId, e.deltaPos);
  }
}

void EventHandler::handle(EntityLookAndRelativeMoveEvent e) {
  optional<Mob> mob = gameState_->getMob(e.entityId);
  if (mob) {
    mob->pos = mob->pos + e.deltaPos;
    mob->look = e.look;
    gameState_->setMob(*mob);
  } else {
    gameState_->setPlayerDeltaPos(e.entityId, e.deltaPos);
    gameState_->setPlayerLook(e.entityId, e.look);
  }
}

void EventHandler::handle(EntityHeadLookEvent e) {
  // N.B. this treats head yaw the same as yaw, which may not be what we want?
  // Not a ton of documentation here...
  gameState_->setPlayerYaw(e.entityId, e.yaw);
}

void EventHandler::handle(DestroyEntitiesEvent e) {
  LOG(INFO) << "-- Destroy [" << e.count << "] entities --";
  int cnt = 0;
  for (uint64_t eid : e.entityIds) {
    if (gameState_->deleteObject(eid) == 1 && gameState_->deleteItemStack(eid) == 1) {
      LOG(INFO) << "[" << cnt << "] Removed entity from client memory with id: " << eid;
    }
    cnt++;
  }
}

void EventHandler::handle(EntityTeleportEvent e) {
  optional<Mob> mob = gameState_->getMob(e.entityId);
  if (mob) {
    mob->pos = e.pos;
    gameState_->setMob(*mob);
  } else {
    gameState_->setPlayerPosAndLook(e.entityId, e.pos, e.look);
  }
}

void EventHandler::handle(WindowItemsEvent e) {
  LOG(INFO) << "Received window items id=" << (int)e.windowId << " #slots=" << e.slots.size();

  CHECK_EQ(e.windowId, gameState_->getCurrentOpenWindowId())
      << "Received window items for not open window";

  if (e.windowId == 0) {
    gameState_->setPlayerInventory(e.slots);
  } else {
    gameState_->setOpenWindowItems(e.slots);
  }

  if (openWindowCondition_ || setSlotCondition_) {
    triggerCondition();
  }
}

void EventHandler::handle(SetSlotEvent e) {
  if (e.windowId == 0) {
    gameState_->setPlayerInventorySlot(e.index, e.slot);
  } else if (e.windowId == gameState_->getCurrentOpenWindowId()) {
    gameState_->setOpenWindowSlot(e.index, e.slot);
  } else {
    LOG(FATAL) << "Can't handle SetSlotEvent with id=" << (int)e.windowId
               << ", open window id=" << gameState_->getCurrentOpenWindowId();
  }
  if (setSlotCondition_) {
    triggerCondition();
  }
}

void EventHandler::handle(ServerDifficultyEvent e) {
  LOG(INFO) << "Received server difficulty: " << (int)e.difficulty;
}

void EventHandler::handle(SpawnMobEvent e) {
  Mob mob = {e.uuid, e.entityId, e.mobType, e.pos, e.look};
  gameState_->setMob(mob);
}

void EventHandler::handle(SpawnObjectEvent e) {
  Object object = {e.uuid, e.entityId, e.objectType, e.pos};
  gameState_->setObject(object);
}

// ItemStack is a special type of Object
void EventHandler::handle(SpawnItemStackEvent e) {
  optional<Object> object = gameState_->getObject(e.entityId);
  if (object) {
    LOG(INFO) << "Update Item Stack Metadata, eid:" << e.entityId << ", item: " << e.item
              << ", pos: " << object->pos;
    ItemStack itemStack = {object->uuid, e.entityId, e.item, object->pos};
    gameState_->setItemStack(itemStack);
  }
}

void EventHandler::handle(UpdateHealthEvent e) {
  gameState_->setHealth(e.health);
  gameState_->setFoodLevel(e.foodLevel);
}

void EventHandler::handle(TimeUpdateEvent e) {
  gameState_->setWorldAge(e.worldAge);
  gameState_->setTimeOfDay(e.timeOfDay);
}

void EventHandler::handle(CollectItemEvent e) {
  // count delta is negative
  gameState_->setItemStackDeltaCount(e.collectedEntityId, -e.count);
}

void EventHandler::handle(OpenWindowEvent e) {
  LOG(INFO) << "Set open window id=" << (int)e.windowId << " type=" << (int)e.windowType;
  gameState_->setCurrentOpenWindow(e.windowId, e.windowType);
}

void EventHandler::handle(ConfirmTransactionEvent e) {
  CHECK(e.accepted) << "window click failed id=" << (int)e.windowId
                    << " counter=" << (int)e.counter;
  CHECK(setSlotCondition_) << "Unexpected confirm transaction packet";
  triggerCondition();
}

void EventHandler::handle(EntityEquipmentEvent e) {
  if (e.which == 0) {  // main hand
    gameState_->setPlayerMainHand(e.entityId, e.slot);
  }
}

/////////////////////
// Condition variable
/////////////////////

void EventHandler::setBlockChangeCondition(BlockPos blockPos) {
  condition_.clear();
  blockChangeCondition_ = blockPos;
}

void EventHandler::setOpenWindowCondition() {
  condition_.clear();
  openWindowCondition_ = true;
}

void EventHandler::setSetSlotCondition() {
  condition_.clear();
  setSlotCondition_ = true;
}

void EventHandler::waitForCondition() { condition_.wait(); }

void EventHandler::triggerCondition() {
  blockChangeCondition_ = nullopt;
  openWindowCondition_ = false;
  setSlotCondition_ = false;

  condition_.trigger();
}
