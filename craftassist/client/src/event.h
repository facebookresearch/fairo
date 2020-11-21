// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <array>
#include <string>
#include "types.h"

////////////////
// Events
//
// Each event represents a clientbound packet that requires
// handling (a response, or a modification of GameState). These
// events are created by PacketReader and sent to EventHandler.

struct KeepaliveEvent {
  uint64_t id;
};

struct LoginSuccessEvent {
  std::string uuid;
  std::string username;
};

struct JoinGameEvent {
  uint32_t entityId;
  GameMode gameMode;
};

struct SpawnPositionEvent {
  BlockPos pos;
};

struct ChunkDataEvent {
  int cx;
  int cz;
  std::array<ChunkSection, 16> chunks;
};

struct PlayerPositionAndLookEvent {
  Pos pos;
  Look look;
  uint8_t flags;
  long teleportId;
};

struct BlockChangeEvent {
  BlockPos pos;
  Block block;
};

struct ChatMessageEvent {
  std::string chat;
  uint8_t position;
};

struct AddPlayersEvent {
  std::vector<std::pair<std::string, std::string>> uuidNamePairs;
};

struct RemovePlayersEvent {
  std::vector<std::string> uuidsToRemove;
};

struct SpawnPlayerEvent {
  unsigned long entityId;
  std::string uuid;
  Pos pos;
  Look look;
};

struct EntityRelativeMoveEvent {
  unsigned long entityId;
  Pos deltaPos;
};

struct EntityLookAndRelativeMoveEvent {
  unsigned long entityId;
  Pos deltaPos;
  Look look;
};

struct EntityTeleportEvent {
  unsigned long entityId;
  Pos pos;
  Look look;
};

struct EntityHeadLookEvent {
  unsigned long entityId;
  float yaw;
};

struct DestroyEntitiesEvent {
  unsigned long count;
  std::vector<uint64_t> entityIds;
};

struct WindowItemsEvent {
  uint8_t windowId;
  std::vector<Slot> slots;
};

struct SetSlotEvent {
  uint8_t windowId;
  uint16_t index;
  Slot slot;
};

struct ServerDifficultyEvent {
  uint8_t difficulty;
};

struct SpawnMobEvent {
  unsigned long entityId;
  std::string uuid;
  uint8_t mobType;
  Pos pos;
  Look look;
};

struct SpawnObjectEvent {
  uint64_t entityId;
  std::string uuid;
  uint8_t objectType;
  Pos pos;
};

struct SpawnItemStackEvent {
  uint64_t entityId;
  Slot item;
};

struct CollectItemEvent {
  uint64_t collectedEntityId;
  uint64_t collectorEntityId;
  uint8_t count;
};

struct UpdateHealthEvent {
  float health;
  uint32_t foodLevel;
};

struct TimeUpdateEvent {
  long worldAge;
  long timeOfDay;
};

struct OpenWindowEvent {
  uint8_t windowId;
  WindowType windowType;
};

struct ConfirmTransactionEvent {
  uint8_t windowId;
  uint16_t counter;
  bool accepted;
};

struct EntityEquipmentEvent {
  unsigned long entityId;
  uint8_t which;
  Slot slot;
};
