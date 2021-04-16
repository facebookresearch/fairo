// Copyright (c) Facebook, Inc. and its affiliates.

#include "game_state.h"
#include <glog/logging.h>
#include "block_map.h"
#include "types.h"

using namespace std;
using std::optional;

void GameState::setPosition(BlockPos pos) {
  // Set float position to center of block
  player_.pos.x = double(pos.x) + 0.5;
  player_.pos.y = double(pos.y);
  player_.pos.z = double(pos.z) + 0.5;
}

void GameState::addChat(const string& chat) { chatHistory_.push_back(chat); }

void GameState::addPlayer(const string& uuid, const string& name) {
  otherPlayerNames_[uuid] = name;
}

void GameState::setPlayer(const string& uuid, unsigned long entityId, Pos pos, Look look) {
  Player player;
  player.uuid = uuid;
  player.entityId = entityId;
  player.pos = pos;
  player.look = look;
  player.mainHand = EMPTY_SLOT;
  player.name = otherPlayerNames_[uuid];

  otherPlayers_[uuid] = player;
}

void GameState::setPlayerDeltaPos(unsigned long entityId, Pos deltaPos) {
  optional<Player> p = getOtherPlayerByEntityId(entityId);
  if (!p) {
    return;
  }
  p->pos = p->pos + deltaPos;
  setPlayer(p->uuid, *p);
}

void GameState::setPlayerPos(unsigned long entityId, Pos pos) {
  optional<Player> p = getOtherPlayerByEntityId(entityId);
  if (!p) {
    return;
  }
  p->pos = pos;
  setPlayer(p->uuid, *p);
}

void GameState::setPlayerPosAndLook(unsigned long entityId, Pos pos, Look look) {
  optional<Player> p = getOtherPlayerByEntityId(entityId);
  if (!p) {
    return;
  }
  p->pos = pos;
  p->look = look;
  setPlayer(p->uuid, *p);
}

void GameState::setPlayerLook(unsigned long entityId, Look look) {
  optional<Player> p = getOtherPlayerByEntityId(entityId);
  if (!p) {
    return;
  }
  p->look = look;
  setPlayer(p->uuid, *p);
}

void GameState::setPlayerYaw(unsigned long entityId, float yaw) {
  optional<Player> p = getOtherPlayerByEntityId(entityId);
  if (!p) {
    return;
  }
  p->look.yaw = yaw;
  setPlayer(p->uuid, *p);
}

void GameState::setPlayerMainHand(unsigned long entityId, Slot slot) {
  optional<Player> p = getOtherPlayerByEntityId(entityId);
  if (!p) {
    return;
  }
  p->mainHand = slot;
  setPlayer(p->uuid, *p);
}

void GameState::removePlayer(const string& uuid) { otherPlayers_.erase(uuid); }

void GameState::removePlayer(unsigned long eid) {
  optional<Player> player = getOtherPlayerByEntityId(eid);
  CHECK(player);
  removePlayer(player->uuid);
}

vector<Player> GameState::getOtherPlayers() {
  vector<Player> players;
  players.reserve(otherPlayers_.size());
  for (pair<string, Player> p : otherPlayers_) {
    players.push_back(p.second);
  }
  return players;
}

void GameState::setMob(Mob mob) { mobs_[mob.entityId] = mob; }

optional<Mob> GameState::getMob(unsigned long entityId) {
  auto mobIter = mobs_.find(entityId);
  if (mobIter == mobs_.end()) {
    return std::nullopt;
  } else {
    return mobIter->second;
  }
}

vector<Mob> GameState::getMobs() {
  vector<Mob> mobs;
  mobs.reserve(mobs_.size());
  for (pair<unsigned long, Mob> p : mobs_) {
    mobs.push_back(p.second);
  }
  return mobs;
}

void GameState::setObject(Object object) { objects_[object.entityId] = object; }

// uint8_t GameState::deleteObject(unsigned long entityId) {LOG(INFO) << "objects_" <<
// objects_.size();printObjectMap();objects_.erase(entityId);LOG(INFO) <<
// objects_.size();printObjectMap();return 0;}
uint8_t GameState::deleteObject(unsigned long entityId) { return objects_.erase(entityId); }

optional<Object> GameState::getObject(unsigned long entityId) {
  auto objectIter = objects_.find(entityId);
  if (objectIter == objects_.end()) {
    return std::nullopt;
  } else {
    return objectIter->second;
  }
}

vector<Object> GameState::getObjects() {
  vector<Object> objects;
  objects.reserve(objects_.size());
  for (pair<unsigned long, Object> p : objects_) {
    objects.push_back(p.second);
  }
  return objects;
}

void GameState::setItemStack(ItemStack itemStack) { itemStacks_[itemStack.entityId] = itemStack; }

// uint8_t GameState::deleteItemStack(unsigned long entityId) {LOG(INFO) << "itemStack_" <<
// itemStacks_.size();printItemStackMap();itemStacks_.erase(entityId);LOG(INFO) <<
// itemStacks_.size(); printItemStackMap();return 0; }
uint8_t GameState::deleteItemStack(unsigned long entityId) { return itemStacks_.erase(entityId); }

optional<ItemStack> GameState::getItemStack(unsigned long entityId) {
  auto itemStackIter = itemStacks_.find(entityId);
  if (itemStackIter == itemStacks_.end()) {
    return std::nullopt;
  } else {
    return itemStackIter->second;
  }
}

vector<ItemStack> GameState::getItemStacks() {
  vector<ItemStack> itemStacks;
  itemStacks.reserve(itemStacks_.size());
  for (pair<unsigned long, ItemStack> p : itemStacks_) {
    itemStacks.push_back(p.second);
  }
  return itemStacks;
}

void GameState::setItemStackDeltaCount(unsigned long entityId, uint8_t deltaCount) {
  optional<ItemStack> p = getItemStack(entityId);
  if (!p) {
    return;
  }
  Slot item = p->item;
  item.count += deltaCount;
  p->item = item;
  setItemStack(*p);
}

optional<Player> GameState::getOtherPlayerByName(const string& name) {
  for (pair<string, Player> p : otherPlayers_) {
    if (p.second.name == name) {
      return p.second;
    }
  }
  return std::nullopt;
}

// FIXME: remove optional<> when all entities (incl. mobs) are stored
optional<Player> GameState::getOtherPlayerByEntityId(unsigned long entityId) {
  // FIXME: O(1) lookup
  for (pair<string, Player> p : otherPlayers_) {
    if (p.second.entityId == entityId) {
      return p.second;
    }
  }
  return std::nullopt;
}

void GameState::setCurrentOpenWindow(uint8_t windowId, WindowType windowType) {
  currentOpenWindowId_ = windowId;

  // add window to windowTypes_
  auto it = windowTypes_.find(windowId);
  if (it == windowTypes_.end()) {
    windowTypes_[windowId] = windowType;
  } else {
    CHECK_EQ(it->second, windowType) << "Overwriting window type for id " << (int)windowId
                                     << " from " << it->second << " to " << windowType;
  }
}

void GameState::setOpenWindowItems(const vector<Slot>& slots) {
  currentOpenWindow_.resize(slots.size());
  currentOpenWindow_ = slots;
}

uint64_t GameState::getInventoryItemCount(uint16_t id, uint8_t meta) {
  uint64_t count = 0;
  for (Slot slot : playerInventory_) {
    if (slot.id == id && slot.meta == meta) {
      count += slot.count;
    }
  }
  return count;
}

unordered_map<Item, uint8_t> GameState::getInventoryItemsCounts() {
  unordered_map<Item, uint8_t> m;
  for (Slot slot : playerInventory_) {
    if (slot.id == 0) {
      continue;
    }
    Item item = {slot.id, slot.meta};
    m[item] += slot.count;
  }
  return m;
}

void GameState::setPlayerInventory(std::vector<Slot> slots) {
  playerInventory_ = slots;
  LOG(INFO) << "-- Set player inventory --";
  int i = 0;
  for (Slot slot : playerInventory_) {
    LOG(INFO) << "index: " << i << ", slot: " << slot;
    i++;
  }
}

void GameState::setPlayerInventorySlot(uint16_t idx, Slot slot) {
  playerInventory_[idx] = slot;
  LOG(INFO) << "Set player inventory slot, idx: " << idx << ", slot: " << slot;
}

bool GameState::isItemStackOnGround(unsigned long entityId) {
  if (getItemStack(entityId) != std::nullopt) {
    return true;
  } else {
    return false;
  }
}

void GameState::printObjectMap() {
  for (auto it = objects_.begin(); it != objects_.end(); ++it) {
    LOG(INFO) << it->first;
  }
}

void GameState::printItemStackMap() {
  for (auto it = itemStacks_.begin(); it != itemStacks_.end(); ++it) {
    LOG(INFO) << it->first;
  }
}
