// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "block_map.h"
#include "blocking_queue.h"
#include "types.h"

class GameState {
 public:
  // Player info
  void setName(const std::string& name) { player_.name = name; }
  const std::string& getName() { return player_.name; }
  void setEntityId(uint64_t id) { player_.entityId = id; }
  uint64_t getEntityId() { return player_.entityId; }
  void setHealth(float health) { player_.health = health; }
  float getHealth() { return player_.health; }
  void setFoodLevel(uint32_t foodLevel) { player_.foodLevel = foodLevel; }
  uint32_t getFoodLevel() { return player_.foodLevel; }
  void setPosition(Pos pos) { player_.pos = pos; }
  void setPosition(BlockPos pos);
  Pos getPosition() { return player_.pos; }
  void setLook(Look look) { player_.look = look; }
  Look getLook() { return player_.look; }
  void setUuid(const std::string& uuid) { player_.uuid = uuid; }
  const std::string& getUuid() { return player_.uuid; }
  Player getPlayer() { return player_; }
  void setWorldAge(long age) { worldAge = age; }
  long getWorldAge() { return worldAge; }
  void setTimeOfDay(long time) { timeOfDay = time; }
  long getTimeOfDay() { return timeOfDay; }

  // Block Map
  BlockMap& getBlockMap() { return blockMap_; }

  // changedBlocks_ access
  BlockingQueue<BlockWithPos>& getChangedBlocks() { return changedBlocks_; };

  // Game Mode
  void setGameMode(GameMode m) { gameMode_ = m; }
  bool getGameMode() { return gameMode_; }

  // Chat
  void addChat(const std::string& chat);
  std::vector<std::string>& getChatHistory() { return chatHistory_; }

  // Other players
  void addPlayer(const std::string& uuid, const std::string& name);
  void setPlayer(const std::string& uuid, unsigned long entityId, Pos pos, Look look);
  void setPlayerDeltaPos(unsigned long entityId, Pos deltaPos);
  void setPlayerPos(unsigned long entityId, Pos pos);
  void setPlayerPosAndLook(unsigned long entityId, Pos pos, Look look);
  void setPlayerLook(unsigned long entityId, Look look);
  void setPlayerYaw(unsigned long entityId, float yaw);
  void setPlayerMainHand(unsigned long entityId, Slot slot);
  void removePlayer(const std::string& uuid);
  void removePlayer(unsigned long entityId);
  std::vector<Player> getOtherPlayers();
  std::optional<Player> getOtherPlayerByName(const std::string& name);
  std::optional<Player> getOtherPlayerByEntityId(unsigned long entityId);

  // Mobs
  void setMob(Mob mob);
  std::optional<Mob> getMob(unsigned long entityId);
  std::vector<Mob> getMobs();

  // Objects
  void setObject(Object object);
  std::optional<Object> getObject(unsigned long entityId);
  std::vector<Object> getObjects();
  uint8_t deleteObject(unsigned long entityId);

  // ItemStacks
  void setItemStack(ItemStack itemStack);
  uint8_t deleteItemStack(unsigned long entityId);
  std::optional<ItemStack> getItemStack(unsigned long entityId);
  std::vector<ItemStack> getItemStacks();
  void setItemStackDeltaCount(unsigned long entityId, uint8_t deltaCount);
  bool isItemStackOnGround(unsigned long entityId);

  // Inventory
  void setPlayerInventory(std::vector<Slot> slots);
  void setPlayerInventorySlot(uint16_t idx, Slot slot);
  const std::vector<Slot>& getPlayerInventory() { return playerInventory_; }
  void setCurrentHotbarIndex(uint8_t i) { currentHotbarIndex_ = i; }
  uint8_t getCurrentHotbarIndex() { return currentHotbarIndex_; }
  std::unordered_map<Item, uint8_t> getInventoryItemsCounts();
  uint64_t getInventoryItemCount(uint16_t id, uint8_t meta);

  // Windows
  void setCurrentOpenWindow(uint8_t windowId, WindowType windowType);
  uint8_t getCurrentOpenWindowId() { return currentOpenWindowId_; }
  void setOpenWindowItems(const std::vector<Slot>& slots);
  void setOpenWindowSlot(uint16_t i, Slot slot) { currentOpenWindow_[i] = slot; }
  const std::vector<Slot>& getOpenWindowItems() { return currentOpenWindow_; }

  void printObjectMap();
  void printItemStackMap();

 private:
  void setPlayer(const std::string& uuid, Player p) { otherPlayers_[uuid] = p; }

  // Fields
  Player player_;
  BlockMap blockMap_;
  GameMode gameMode_;
  long worldAge = 0;
  long timeOfDay = 0;
  std::vector<std::string> chatHistory_;
  std::unordered_map<std::string, std::string> otherPlayerNames_;
  std::unordered_map<std::string, Player> otherPlayers_;
  std::unordered_map<unsigned long, Mob> mobs_;
  std::unordered_map<unsigned long, Object> objects_;
  std::unordered_map<unsigned long, ItemStack> itemStacks_;
  std::vector<Slot> playerInventory_;
  uint8_t currentHotbarIndex_ = 0;
  uint8_t currentOpenWindowId_ = 0;
  std::vector<Slot> currentOpenWindow_;
  std::unordered_map<uint8_t, WindowType> windowTypes_;
  BlockingQueue<BlockWithPos> changedBlocks_;
};
