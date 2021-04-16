// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include "condition.h"
#include "encoder.h"
#include "event.h"
#include "game_state.h"
#include "packet_writer.h"
#include "types.h"

// EventHandler receives Events from PacketReader. Its responsibility is to
// respond (via PacketWriter) to conform to the Minecraft protocol, and modify
// the GameState as necessary.
//
// Minecraft protocol: http://wiki.vg/Protocol
class EventHandler {
 public:
  EventHandler() {}
  EventHandler(GameState* gameState, PacketWriter* packetWriter, Encoder* encoder);
  void handle(KeepaliveEvent e);
  void handle(LoginSuccessEvent e);
  void handle(JoinGameEvent e);
  void handle(SpawnPositionEvent e);
  void handle(ChunkDataEvent e);
  void handle(PlayerPositionAndLookEvent e);
  void handle(BlockChangeEvent e);
  void handle(ChatMessageEvent e);
  void handle(AddPlayersEvent e);
  void handle(RemovePlayersEvent e);
  void handle(SpawnPlayerEvent e);
  void handle(EntityRelativeMoveEvent e);
  void handle(EntityLookAndRelativeMoveEvent e);
  void handle(EntityHeadLookEvent e);
  void handle(DestroyEntitiesEvent e);
  void handle(EntityTeleportEvent e);
  void handle(WindowItemsEvent e);
  void handle(SetSlotEvent e);
  void handle(ServerDifficultyEvent e);
  void handle(SpawnMobEvent e);
  void handle(SpawnObjectEvent e);
  void handle(SpawnItemStackEvent e);
  void handle(UpdateHealthEvent e);
  void handle(TimeUpdateEvent e);
  void handle(CollectItemEvent e);
  void handle(OpenWindowEvent e);
  void handle(ConfirmTransactionEvent e);
  void handle(EntityEquipmentEvent e);

  // Condition variable methods, to be called by other threads to synchronize
  // with the event handler thread.

  // Prepare to wait for a block change event at blockPos
  void setBlockChangeCondition(BlockPos blockPos);

  // Prepare to wait for an open window + window items
  void setOpenWindowCondition();

  // Prepare to wait for a set slot, confirm transaction, or window items packet
  void setSetSlotCondition();

  // Block until the condition is triggered
  void waitForCondition();

 private:
  // Notify waiting thread that the condition is triggered
  void triggerCondition();

  // Fields
  GameState* gameState_;
  PacketWriter* packetWriter_;
  Encoder* encoder_;
  Condition condition_;
  int chunkDataEvents_ = 0;

  // TODO: better interface for wait conditions
  std::optional<BlockPos> blockChangeCondition_;
  bool openWindowCondition_ = false;
  bool setSlotCondition_ = false;
};
