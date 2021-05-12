// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <glog/logging.h>
#include <string>
#include <vector>

#include "condition.h"
#include "encoder.h"
#include "event.h"
#include "event_handler.h"
#include "game_state.h"
#include "packet_reader.h"
#include "packet_writer.h"

class Client {
  const long STEP_COOLDOWN_MS = 125;
  const uint16_t HELD_ITEM_IDX = 36;  // see player inventory diagram
  const long TICK_PER_SEC = 20;  // MC time is based on ticks, where 20 ticks happen every second

 public:
  static constexpr double HEIGHT = 1.5;

  Client(const std::string& host, int port, const std::string& username);
  ~Client() { disconnect(); }
  void disconnect();
  Pos getPosition() { return gameState_->getPosition(); }
  const std::string& getName() { return gameState_->getName(); }
  Player getPlayer() { return gameState_->getPlayer(); }
  bool isCreativeMode() { return gameState_->getGameMode() == GameMode::CREATIVE; }

  // Get block-id and depth vision from current pos and look
  void getVision(std::vector<Block>& blocks, std::vector<float>& depth, int height, int width,
                 int maxDepth);

  // Get the block intersected by the client's direct line of sight, limited to maxDepth blocks.
  //
  // Returns true and sets *block, *depth, and *blockPos iff there is a valid intersection.
  bool getLineOfSight(Block* block, float* depth, BlockPos* blockPos, int maxDepth);

  // Get the block intersected by another player's direct line of sight, limited to maxDepth blocks.
  //
  // Returns true and sets *block, *depth, and *blockPos iff there is a valid intersection.
  bool getPlayerLineOfSight(Block* block, float* depth, BlockPos* blockPos, Player player,
                            int maxDepth);

  // Get a cuboid of blocks with the given boundaries (inclusive)
  // The result is given in (y, z, x) order
  void getCuboid(std::vector<Block>& v, int xa, int xb, int ya, int yb, int za, int zb);

  // Get a cuboid of blocks with radii (rx, ry, rz) centered around the
  // player's BlockPos.
  void getLocalCuboid(std::vector<Block>& ob, int rx, int ry, int rz);

  // Return a list of all chat strings received to date
  const std::vector<std::string>& getChatHistory();

  // Time is based on ticks, where 20 ticks happen every second (default).
  // There are 24000 ticks in a day, making Minecraft days exactly 20 minutes long.
  // Always incrementing
  long getWorldAge() { return gameState_->getWorldAge(); }

  // The time of day is based on the timestamp modulo 24000 (default).
  // 0 is sunrise, 6000 is noon, 12000 is sunset, and 18000 is midnight.
  long getTimeOfDay() { return gameState_->getTimeOfDay(); }

  // Attempts a step in the (x, z) plane. It will step up or fall down a
  // single block if necessary, otherwise movement is prohibited. i.e.
  // its y-position can change by at most 1 per step.
  //
  // N.B. this does not allow movement across a deep chasm one width
  // across, which is possible by jumping.
  void discreteStep(int dx, int dz);

  // Attempts a step in the (x, y, z), without considering gravity
  void discreteFly(int dx, int dy, int dz);

  // Attempts a step in the (x, z) plane in the direction of player's current yaw
  void discreteStepForward();

  // Send the given string to the world as chat
  void sendChat(const std::string& s);

  // Return a list of all other players in the map
  std::vector<Player> getOtherPlayers();

  // Return a list of all mobs in the map, with their *absolute* positions
  std::vector<Mob> getMobs();

  // Return a list of all item stacks in the map, with their *absolute* positions
  std::vector<ItemStack> getItemStacks();

  std::optional<ItemStack> getItemStack(unsigned long entityId) {
    return gameState_->getItemStack(entityId);
  }

  // Check if a given item stack is on the ground
  bool isItemStackOnGround(uint64_t entityId);

  // Return the Player struct for the named player
  std::optional<Player> getOtherPlayerByName(const std::string& name);

  // Return a list of 9 hotbar slots, which are slots 36-44 of the player inventory
  std::vector<Slot> getPlayerHotbar();

  // Set held item. Returns true if sucessful, false if item not in inventory.
  // Always returns true in creative mode.
  bool setHeldItem(Item item);

  // Get item at current hotbar slot
  Slot getCurrentHotbarItem();

  // Place currently-held block
  // Return true if successful, false otherwise (e.g. if the block was too far)
  // placeBlockFace() places the block in front of the agent's face
  // placeBlockFeet() places the block in front of the agent's feet
  bool placeBlock(BlockPos pos);
  bool placeBlockFace();
  bool placeBlockFeet();

  // Use currently-held item on a target entity (e.g. a block)
  // Return true if successful, false otherwise (e.g. if the target block was too far)
  bool useEntity(BlockPos pos);

  // Use currently-held item
  // Return true if successful, false otherwise
  bool useItem();

  // Use the currently held item on a target block
  // Return true if successful, false otherwise
  bool useItemOnBlock(BlockPos pos);

  // Dig a block to completion.
  // Return true if successful, false otherwise (e.g. if the block was too far)
  // digFace() digs the block in front of the agent's face
  // digFeet() digs the block in front of the agent's face
  // TODO: remove digFace/digFeet, add digRelative
  bool dig(BlockPos pos);
  bool digFace();
  bool digFeet();

  // Drop the selected item (stacks)
  // dropItemStackInHand() drops the entire stack in hand, while
  // dropItemInHand() only drops the currently selected item
  void dropItemStackInHand();
  void dropItemInHand();
  bool dropInventoryItemStack(uint16_t id, uint8_t meta, uint8_t count);

  void setInventorySlot(int16_t index, uint16_t id, uint8_t meta, uint8_t count);

  const std::vector<Slot>& getPlayerInventory() { return gameState_->getPlayerInventory(); }
  std::unordered_map<Item, uint8_t> getInventoryItemsCounts() {
    return gameState_->getInventoryItemsCounts();
  }

  uint64_t getInventoryItemCount(uint16_t id, uint8_t meta) {
    return gameState_->getInventoryItemCount(id, meta);
  }

  // Turn counter-clockwise by `angle` degrees
  void turnAngle(float angle);

  // Turn left or right 90 degrees
  void turnLeft();
  void turnRight();

  // Set absolute yaw/pitch
  void setLook(Look look);

  // Craft the item with given id/meta
  //
  // If successful, returns the number of items crafted
  // If unsucessful (e.g. ingredients not found, no crafting table), returns negative
  int craft(uint16_t id, uint8_t meta);

  // Return a list of blocks that have changed since last call to this fn
  std::vector<BlockWithPos> getChangedBlocks() { return gameState_->getChangedBlocks().popall(); }

 private:
  void socketConnect(const std::string& host, int port);

  // Move to newp without doing any checks to make sure you're not
  // walking on air or through blocks. This method should ideally be used
  // by a method that performs these checks.
  void doMoveUnsafe(Pos newp);

  // Crafting

  // Open any nearby crafting table
  // Returns the window id of the table, or -1 if no nearby table is found
  int openNearbyCraftingTable();

  // Returns the position of any reachable crafting table
  std::optional<BlockPos> findNearbyCraftingTable();

  // Perform a single click in the given window (inventory = 0)
  void windowClick(uint16_t windowId, uint16_t idx, bool rightClick);

  // Move a single item from slot idx `from` to slot idx `to`
  void windowMoveItem(uint16_t windowId, uint16_t from, uint16_t to);

  // Swap the item stacks at indices a and b
  void windowSwapSlots(uint16_t windowId, uint16_t a, uint16_t b);

  // Return the index of the item in the crafting window, or -1 if not found
  int craftWindowFindIngredient(uint16_t id, uint8_t meta);

  const std::vector<Slot>& getOpenWindowItems() { return gameState_->getOpenWindowItems(); }

  // Fields
  bool disconnected_ = false;
  std::unique_ptr<Encoder> encoder_;
  std::unique_ptr<GameState> gameState_;
  std::unique_ptr<EventHandler> eventHandler_;
  std::unique_ptr<PacketWriter> packetWriter_;
  std::unique_ptr<PacketReader> packetReader_;
  int socket_;
  std::thread packetReaderThread_;
  Condition loaded_;
  std::unordered_map<uint8_t, uint16_t> windowCounters;
};
