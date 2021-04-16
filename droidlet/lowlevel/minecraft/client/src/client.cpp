// Copyright (c) Facebook, Inc. and its affiliates.

#include <errno.h>
#include <glog/logging.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <optional>

#include "block_data.h"
#include "client.h"
#include "craft_recipes.h"
#include "encoder.h"
#include "event.h"
#include "event_handler.h"
#include "game_state.h"
#include "graphics.h"
#include "packet_reader.h"
#include "packet_writer.h"
#include "util.h"

using namespace std;
using std::min;
using std::nullopt;
using std::optional;

bool IS_GLOG_INITIALIZED = false;

////////////////
// Public

Client::Client(const string& host, int port, const string& username) {
  // Init glog. Better place for this?
  if (!IS_GLOG_INITIALIZED) {
    google::InitGoogleLogging("client");
    google::InstallFailureSignalHandler();
    IS_GLOG_INITIALIZED = true;
  }

  // Connect to Minecraft server
  socketConnect(host, port);

  // Init fields / threads
  encoder_ = unique_ptr<Encoder>(new Encoder());
  gameState_ = unique_ptr<GameState>(new GameState());
  packetWriter_ = unique_ptr<PacketWriter>(new PacketWriter(socket_));
  eventHandler_ = unique_ptr<EventHandler>(
      new EventHandler(gameState_.get(), packetWriter_.get(), encoder_.get()));
  packetReader_ = unique_ptr<PacketReader>(new PacketReader(socket_, eventHandler_.get()));
  packetReaderThread_ = packetReader_->startThread();

  // Do login
  LOG(INFO) << "Writing login packets";
  packetWriter_->safeWrite(encoder_->handshakePacket(host, port, false));
  packetWriter_->safeWrite(encoder_->loginStartPacket(username));

  // Wait until chunks have loaded
  eventHandler_->waitForCondition();
}

void Client::disconnect() {
  if (disconnected_) {
    return;
  }
  LOG(INFO) << "Reaping socket " << socket_;
  close(socket_);
  LOG(INFO) << "Closing PacketReader thread";
  packetReaderThread_.join();
  LOG(INFO) << "Closed PacketReader thread";
  disconnected_ = true;
}

void Client::discreteStep(int dx, int dz) {
  Pos p = gameState_->getPosition();
  Pos newp = p + BlockPos{dx, 0, dz};
  if (!gameState_->getBlockMap().isBlockLoaded(newp.toBlockPos())) {
    LOG(WARNING) << "Trying to step to unloaded pos: " << newp;
    return;
  }
  if (gameState_->getBlockMap().canStandAt(newp)) {
    // Walk along flat ground
    doMoveUnsafe(newp);
  } else if (gameState_->getBlockMap().canStandAt(newp + BlockPos{0, 1, 0})) {
    // Climb one step
    doMoveUnsafe(newp + BlockPos{0, 1, 0});
  } else if (gameState_->getBlockMap().canStandAt(newp + BlockPos{0, -1, 0})) {
    // Drop one step
    doMoveUnsafe(newp + BlockPos{0, -1, 0});
  }
  // Otherwise, take no move
}

void Client::discreteFly(int dx, int dy, int dz) {
  Pos p = gameState_->getPosition();
  Pos newp = p + BlockPos{dx, dy, dz};
  if (!gameState_->getBlockMap().isBlockLoaded(newp.toBlockPos())) {
    LOG(WARNING) << "Trying to step to unloaded pos: " << newp;
    return;
  }
  if (gameState_->getBlockMap().canWalkthrough(newp)) {
    doMoveUnsafe(newp);
  }
}

void Client::discreteStepForward() {
  float yaw = gameState_->getLook().yaw;
  BlockPos d = discreteStepDirection(yaw);
  discreteStep(d.x, d.z);
}

void Client::getVision(vector<Block>& blocks, vector<float>& depth, int height, int width,
                       int maxDepth) {
  vector<BlockPos> blockPos;
  Graphics::vision(blocks, depth, blockPos, height, width, gameState_->getBlockMap(),
                   gameState_->getPosition(), gameState_->getPlayer().look, maxDepth);
}

bool Client::getLineOfSight(Block* block, float* depth, BlockPos* blockPos, int maxDepth) {
  return Graphics::lineOfSight(block, depth, blockPos, gameState_->getBlockMap(),
                               gameState_->getPosition(), gameState_->getPlayer().look, maxDepth);
}

bool Client::getPlayerLineOfSight(Block* block, float* depth, BlockPos* blockPos, Player player,
                                  int maxDepth) {
  return Graphics::lineOfSight(block, depth, blockPos, gameState_->getBlockMap(), player.pos,
                               player.look, maxDepth);
}

void Client::getCuboid(vector<Block>& v, int xa, int xb, int ya, int yb, int za, int zb) {
  gameState_->getBlockMap().getCuboid(v, xa, xb, ya, yb, za, zb);
}

void Client::getLocalCuboid(vector<Block>& v, int rx, int ry, int rz) {
  BlockPos p = getPosition().toBlockPos();
  getCuboid(v, p.x - rx, p.x + rx, p.y - ry, p.y + ry, p.z - rz, p.z + rz);
}

const vector<string>& Client::getChatHistory() { return gameState_->getChatHistory(); }

void Client::sendChat(const string& s) { packetWriter_->safeWrite(encoder_->chatMessagePacket(s)); }

vector<Player> Client::getOtherPlayers() { return gameState_->getOtherPlayers(); }

vector<Mob> Client::getMobs() { return gameState_->getMobs(); }

vector<ItemStack> Client::getItemStacks() { return gameState_->getItemStacks(); }

optional<Player> Client::getOtherPlayerByName(const string& name) {
  return gameState_->getOtherPlayerByName(name);
}

vector<Slot> Client::getPlayerHotbar() {
  const vector<Slot>& inventory = gameState_->getPlayerInventory();
  vector<Slot> hotbar = vector<Slot>(&inventory[36], &inventory[45]);
  return hotbar;
}

bool Client::setHeldItem(Item item) {
  if (gameState_->getGameMode() == GameMode::CREATIVE) {
    Slot slot = {item.id, item.meta, 1, 0};
    packetWriter_->safeWrite(encoder_->creativeInventoryActionPacket(HELD_ITEM_IDX, slot));
    gameState_->setPlayerInventorySlot(HELD_ITEM_IDX, slot);  // optimistic update
    return true;
  } else {
    // survival
    for (size_t i = 0; i < gameState_->getPlayerInventory().size(); i++) {
      Slot slot = gameState_->getPlayerInventory()[i];
      if (slot.id == item.id && slot.meta == item.meta) {
        if (i != HELD_ITEM_IDX) {
          windowSwapSlots(0, i, HELD_ITEM_IDX);
        }
        packetWriter_->safeWrite(encoder_->closeWindowPacket(0));
        return true;
      }
    }
    return false;
  }
}

Slot Client::getCurrentHotbarItem() {
  uint8_t index = gameState_->getCurrentHotbarIndex() + 36;  // hotbar starts at idx 36
  return gameState_->getPlayerInventory()[index];
}

bool Client::placeBlock(BlockPos pos) {
  Slot heldItem = getCurrentHotbarItem();
  CHECK_GE(heldItem.id, 1) << "Can't place block with id: " << heldItem.id;

  Pos myPos = getPosition();
  Pos dist = myPos - pos;

  if (myPos.toBlockPos() == pos || myPos.toBlockPos() + BlockPos{0, 1, 0} == pos) {
    LOG(WARNING) << "Can't place where I'm standing. Tried to place on " << pos << " from "
                 << myPos;
    return false;
  }
  if (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z > 16) {
    LOG(WARNING) << "Cannot place farther than 4-block radius. Tried to place on" << pos << " from "
                 << myPos;
    return false;
  }

  // The locking/ordering here matters to avoid hanging indefinitely!
  // There is a possible race between us and another player placing a block at
  // the same position. We only get block change events, and not their origin.
  // We wish to avoid this situation:
  // 1. check pos is empty
  // 2. read someone else's block change at pos (packet reader thread)
  // 3. set block change condition
  // 4. our block place fails silently because there is already a block
  // 5. wait indefinitely for a block change that will never come
  //
  // We avoid this by doing step 1 and step 3 holding the block map lock, which
  // forces step 2 to occur before (then pos is not empty at step 1) or after
  // (then the block change condition will be triggered).
  //
  // FIXME: check if entity exists in target block, and return false!
  gameState_->getBlockMap().lock();
  optional<Block> block = gameState_->getBlockMap().getBlockUnsafe(pos.x, pos.y, pos.z);
  CHECK(block) << "Failed to place block at unloaded " << pos;
  if (*block != BLOCK_AIR) {
    gameState_->getBlockMap().unlock();
    LOG(WARNING) << "Can't place block at occupied " << pos << " -> " << *block;
    return false;
  }
  eventHandler_->setBlockChangeCondition(pos);
  gameState_->getBlockMap().unlock();

  packetWriter_->safeWrite(encoder_->playerBlockPlacementPacket(pos));
  if (heldItem.id < 256) {  // don't wait for mobs
    eventHandler_->waitForCondition();
  }

  return true;
}

// TODO Add more checkings
bool Client::useEntity(BlockPos pos) {
  Slot heldItem = getCurrentHotbarItem();
  CHECK_GE(heldItem.id, 1) << "Can't use item with id on target: " << heldItem.id;

  Pos myPos = getPosition();
  Pos dist = myPos - pos;

  if (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z > 16) {
    LOG(WARNING) << "Cannot use entity farther than 4-block radius. Tried to use entity on" << pos
                 << " from " << myPos;
    return false;
  }

  packetWriter_->safeWrite(encoder_->playerUseEntityPacket(pos));

  return true;
}

// TODO Add more checkings
bool Client::useItem() {
  Slot heldItem = getCurrentHotbarItem();
  CHECK_GE(heldItem.id, 1) << "Can't use item with id: " << heldItem.id;

  packetWriter_->safeWrite(encoder_->playerUseItemPacket());

  return true;
}

bool Client::useItemOnBlock(BlockPos pos) {
  Slot heldItem = getCurrentHotbarItem();
  CHECK_GE(heldItem.id, 1) << "Can't place block with id: " << heldItem.id;

  Pos myPos = getPosition();
  Pos dist = myPos - pos;

  if (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z > 16) {
    LOG(WARNING) << "Cannot place farther than 4-block radius. Tried to place on" << pos << " from "
                 << myPos;
    return false;
  }

  gameState_->getBlockMap().lock();
  optional<Block> block = gameState_->getBlockMap().getBlockUnsafe(pos.x, pos.y, pos.z);
  CHECK(block) << "Failed to use item on block at unloaded " << pos;
  gameState_->getBlockMap().unlock();

  // Use item on block is equivalent to holding the item and then placing it on the position of
  // the target (at least for fertilizing tree sapling with bone meal)
  packetWriter_->safeWrite(encoder_->playerBlockPlacementPacket(pos));

  return true;
}

bool Client::placeBlockFeet() {
  float yaw = gameState_->getLook().yaw;
  BlockPos pos = getPosition().toBlockPos() + discreteStepDirection(yaw);
  return placeBlock(pos);
}

bool Client::placeBlockFace() {
  float yaw = gameState_->getLook().yaw;
  BlockPos pos = getPosition().toBlockPos() + discreteStepDirection(yaw) + BlockPos{0, 1, 0};
  return placeBlock(pos);
}

bool Client::dig(BlockPos pos) {
  if (gameState_->getGameMode() != GameMode::CREATIVE) {
    LOG(FATAL) << "Client::dig currently only valid in creative mode";
  }
  Pos myPos = getPosition();
  Pos dist = myPos - pos;
  if (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z > 16) {
    LOG(WARNING) << "Cannot dig farther than 4-block radius. Tried to dig " << pos << " from "
                 << myPos;
    return false;
  }

  // See comment in placeBlock() for an explanation
  gameState_->getBlockMap().lock();
  optional<Block> block = gameState_->getBlockMap().getBlockUnsafe(pos.x, pos.y, pos.z);
  CHECK(block) << "Failed to dig block at unloaded " << pos;
  if (*block == BLOCK_AIR) {
    gameState_->getBlockMap().unlock();
    LOG(WARNING) << "Can't dig block at empty " << pos;
    return false;
  }
  eventHandler_->setBlockChangeCondition(pos);
  gameState_->getBlockMap().unlock();

  packetWriter_->safeWrite(encoder_->playerStartDiggingPacket(pos));
  eventHandler_->waitForCondition();
  return true;
}

bool Client::digFace() {
  float yaw = gameState_->getLook().yaw;
  BlockPos pos = getPosition().toBlockPos() + discreteStepDirection(yaw) + BlockPos{0, 1, 0};
  return dig(pos);
}

bool Client::digFeet() {
  float yaw = gameState_->getLook().yaw;
  BlockPos pos = getPosition().toBlockPos() + discreteStepDirection(yaw);
  return dig(pos);
}

void Client::dropItemStackInHand() {
  packetWriter_->safeWrite(encoder_->playerDropItemStackInHandPacket());
}

void Client::dropItemInHand() { packetWriter_->safeWrite(encoder_->playerDropItemInHandPacket()); }

bool Client::dropInventoryItemStack(uint16_t id, uint8_t meta, uint8_t count) {
  if (gameState_->getInventoryItemCount(id, meta) < count) {
    return false;
  }

  for (int i = gameState_->getPlayerInventory().size(); i >= 0; i--) {
    Slot slot = gameState_->getPlayerInventory()[i];
    if (slot.id == id && slot.meta == meta) {
      uint8_t delta = min(slot.count, count);
      slot.count = slot.count - delta;
      // set updated inventory slot first
      gameState_->setPlayerInventorySlot(i, slot);
      packetWriter_->safeWrite(encoder_->playerSetInventorySlotPacket(i, slot));
      // drop item stack onto the ground
      slot.count = delta;
      packetWriter_->safeWrite(encoder_->playerDropItemStackPacket(slot));
      count -= delta;
    }
    if (count == 0) {
      return true;
    }
  }
  return false;
}

void Client::setInventorySlot(int16_t index, uint16_t id, uint8_t meta, uint8_t count) {
  Slot slot = {id, meta, count, 0};
  packetWriter_->safeWrite(encoder_->playerSetInventorySlotPacket(index, slot));
}

bool Client::isItemStackOnGround(uint64_t entityId) {
  return gameState_->isItemStackOnGround(entityId);
}

void Client::turnAngle(float angle) {
  Look look = gameState_->getLook();
  look.yaw -= angle;
  if (look.yaw < 0) {
    look.yaw += 360;
  }
  if (look.yaw > 360) {
    look.yaw -= 360;
  }
  setLook(look);
}

void Client::turnLeft() { turnAngle(90); }

void Client::turnRight() { turnAngle(-90); }

void Client::setLook(Look look) {
  gameState_->setLook(look);
  packetWriter_->safeWrite(encoder_->playerLookPacket(look.yaw, look.pitch, true));
}

int Client::craft(uint16_t id, uint8_t meta) {
  int windowId = openNearbyCraftingTable();
  if (windowId < 0) {
    return -1;
  }

  auto recipeIt = RECIPES.find((id << 8) | meta);
  if (recipeIt == RECIPES.end()) {
    LOG(WARNING) << "No recipe for " << (int)id << ":" << (int)meta;
    return -2;
  }
  Recipe recipe = recipeIt->second;
  LOG(INFO) << "Found recipe for " << (int)id << ":" << (int)meta;

  for (int idx = 0; idx < 9; idx++) {
    Ingredient ig = recipe.ingredients[idx];
    if (ig.id == 0) {
      continue;
    }
    for (int i = 0; i < ig.count; i++) {
      int from = craftWindowFindIngredient(ig.id, ig.meta);
      if (from == -1) {
        // item not in inventory
        return -3;
      }

      windowMoveItem(windowId, from, idx + 1);  // + 1 because craft grid is idx 1-9
    }
  }

  // Move result to empty inventory spot
  for (size_t i = 45; i >= 10; i--) {
    if (gameState_->getOpenWindowItems()[i].id == 0) {
      windowSwapSlots(windowId, i, 0);

      // Wait for packet changing inventory slot
      size_t inventoryIdx = i - 1;  // inventory idxs are offset one from crafting table
      while (gameState_->getPlayerInventory()[inventoryIdx].id != id) {
        this_thread::sleep_for(chrono::milliseconds(5));
      }
      break;
    }
  }

  // Close crafting table
  LOG(INFO) << "Closing crafting table";
  packetWriter_->safeWrite(encoder_->closeWindowPacket(windowId));

  return recipe.count;
}

////////////////
// Private

void Client::socketConnect(const string& host, int port) {
  struct hostent* hostEntry = gethostbyname(host.c_str());
  if (!hostEntry) {
    throw runtime_error("gethostbyname failed");
  }

  struct sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(port);
  memcpy(&serverAddr.sin_addr, hostEntry->h_addr_list[0], hostEntry->h_length);

  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (connect(socket_, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
    throw runtime_error("connect failed");
  }
}

void Client::doMoveUnsafe(Pos newp) {
  gameState_->setPosition(newp);
  packetWriter_->safeWrite(encoder_->playerPositionPacket(newp.x, newp.y, newp.z, true));
  this_thread::sleep_for(chrono::milliseconds(STEP_COOLDOWN_MS));
}

optional<BlockPos> Client::findNearbyCraftingTable() {
  BlockPos myPos = getPosition().toBlockPos();
  vector<Block> blocks(7 * 7 * 7);
  getLocalCuboid(blocks, 3, 3, 3);
  for (int ry = 0; ry < 7; ry++) {
    for (int rz = 0; rz < 7; rz++) {
      for (int rx = 0; rx < 7; rx++) {
        if (blocks[ry * 7 * 7 + rz * 7 + rx].id == 58) {
          return BlockPos{myPos.x - 3 + rx, myPos.y - 3 + ry, myPos.z - 3 + rz};
        }
      }
    }
  }
  return nullopt;
}

int Client::openNearbyCraftingTable() {
  optional<BlockPos> pos = findNearbyCraftingTable();
  if (!pos) {
    LOG(INFO) << "No crafting table found near " << getPosition();
    return -1;
  }
  LOG(INFO) << "Opening crafting table at " << *pos;

  // FIXME: race condition here if crafting table is destroyed

  eventHandler_->setOpenWindowCondition();
  packetWriter_->safeWrite(encoder_->playerBlockPlacementPacket(*pos));
  eventHandler_->waitForCondition();
  return gameState_->getCurrentOpenWindowId();
}

void Client::windowClick(uint16_t windowId, uint16_t idx, bool rightClick) {
  Slot slot = (windowId == 0) ? gameState_->getPlayerInventory()[idx]
                              : gameState_->getOpenWindowItems()[idx];

  uint16_t counter = windowCounters[windowId];
  windowCounters[windowId] = ++counter;

  eventHandler_->setSetSlotCondition();
  packetWriter_->safeWrite(encoder_->clickWindowPacket(windowId, idx, rightClick, counter, slot));
  eventHandler_->waitForCondition();
}

void Client::windowMoveItem(uint16_t windowId, uint16_t from, uint16_t to) {
  LOG(INFO) << "windowMoveItem " << (int)from << " -> " << (int)to;
  windowClick(windowId, from, false);
  windowClick(windowId, to, true);
  windowClick(windowId, from, false);
}

void Client::windowSwapSlots(uint16_t windowId, uint16_t a, uint16_t b) {
  LOG(INFO) << "windowSwapSlots " << a << " <-> " << b;
  windowClick(windowId, a, false);
  windowClick(windowId, b, false);
  windowClick(windowId, a, false);
}

int Client::craftWindowFindIngredient(uint16_t id, uint8_t meta) {
  // find ingredients in inventory, not in crafting grid (0-9)
  for (int i = 10; i <= 45; i++) {
    Slot slot = gameState_->getOpenWindowItems()[i];
    if (slot.id == id && slot.meta == meta) {
      return i;
    }
  }
  return -1;
}
