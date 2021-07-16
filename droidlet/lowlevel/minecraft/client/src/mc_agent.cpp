// Copyright (c) Facebook, Inc. and its affiliates.

#include <glog/logging.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <optional>
#include <string>

#include "client.h"
#include "types.h"

using namespace std;
using std::optional;
using std::nullopt;
namespace py = pybind11;

const int VISION_H = 64;
const int VISION_W = 64;
const int VISION_MAX_DEPTH = 64;

class Agent {
 public:
  // Main constructor
  //  - host/port: the Minecraft server to connect to
  //  - username:  username visible to other players on the server. Recommended to be unique.
  Agent(const string& host, int port, const string& username) : client_(host, port, username) {}

  ////////////////
  // Observations
  ////////////////

  // Return a cuboid of blocks between corners (xa, ya, za) and (xb, yb, zb)
  // inclusive.
  //
  // The result is a 4d numpy array in (y, z, x, id/meta) order
  // e.g. result[:,:,:,0] are the block ids
  // e.g. result[0, 0, -1, 1] is the block meta at block pos (ya, za, xb)
  py::array_t<uint8_t> getBlocks(int xa, int xb, int ya, int yb, int za, int zb);
  vector<Mob> getMobs();
  vector<ItemStack> getItemStacks();
  optional<ItemStack> getItemStack(unsigned long entityId) {
    return client_.getItemStack(entityId);
  }
  bool isItemStackOnGround(uint64_t entityId);
  uint64_t getInventoryItemCount(uint16_t id, uint8_t meta);

  // Return a cube of blocks centered around the agent, with radius r, in the
  // same order as getBlocks().
  //
  // e.g. if r == 4, the result is a 9x9x9x2 numpy array where
  // result[5, 5, 5, :] is the block in which the agent resides.
  py::array_t<uint8_t> getLocalBlocks(int r);

  // Return a list of new incoming chats received since the last time this
  // method was called.
  vector<string> getIncomingChats();

  // Return the agent's Player struct.
  //
  // See client/src/types.h
  Player getPlayer() { return client_.getPlayer(); }

  // Return a list of Player structs for every other player
  // (bot or human) in the world.
  vector<Player> getOtherPlayers() { return client_.getOtherPlayers(); }

  // Return the Player struct, searching by its `name` field, or None if no
  // such player exists.
  optional<Player> getOtherPlayerByName(const string& name) {
    return client_.getOtherPlayerByName(name);
  }

  // Return the agent vision as a pair of numpy arrays (blocks, depth), where:
  // blocks has shape (64, 64, 2) and dtype='uint8'
  // depth  has shape (64, 64) and dtype='float32'
  py::tuple getVision();

  // Return the block pos that is directly intersected by the agent's vision, or None
  optional<BlockPos> getLineOfSight();

  // Return the block pos that is directly intersected by another player's vision, or None
  optional<BlockPos> getPlayerLineOfSight(Player player);

  // Return a list of blocks that have changed since the last call of this fn
  vector<py::tuple> getChangedBlocks();

  // Return current world age based on ticks
  long getWorldAge() { return client_.getWorldAge(); }

  // Return time of the day based on ticks
  long getTimeOfDay() { return client_.getTimeOfDay(); }

  ////////////////
  // Actions
  ////////////////

  // Disconnect the agent from the server. Do not call any other method after
  // this one.
  void disconnect() { client_.disconnect(); }

  // Send a "dig" packet for the block at (x, y, z)
  bool dig(int x, int y, int z) { return client_.dig({x, y, z}); }

  // Drop the selected item stack in hand
  void dropItemStackInHand() { client_.dropItemStackInHand(); }

  // Drop the selected item in hand
  void dropItemInHand() { client_.dropItemInHand(); }

  // Drop the selected item stack
  bool dropInventoryItemStack(uint16_t id, uint8_t meta, uint8_t count) {
    return client_.dropInventoryItemStack(id, meta, count);
  }

  void setInventorySlot(int16_t index, uint16_t id, uint8_t meta, uint8_t count) {
    client_.setInventorySlot(index, id, meta, count);
  }

  const std::vector<Slot>& getPlayerInventory() { return client_.getPlayerInventory(); }

  std::unordered_map<Item, uint8_t> getInventoryItemsCounts() {
    return client_.getInventoryItemsCounts();
  }

  // Broadcast the given string to the in-game chat
  void sendChat(const string& s) { client_.sendChat(s); }

  // Change the item the agent is holding in its main hand.
  //
  // arg can be:
  // - an int: interpreted as an item id (e.g. 41 for "gold block")
  // - an (int, int) tuple: interpreted as an (id, meta) pair
  //
  // See https://minecraft-ids.grahamedgecombe.com/
  bool setHeldItem(py::object arg);

  // Take a one-block step in the given direction, and sleep for 250ms to limit
  // the agent to the same average walking speed as a human player.
  //
  // N.B. the y-direction steps are only legal in creative mode (which have no
  // gravity)
  void stepPosX() { doMove(1, 0, 0); }
  void stepNegX() { doMove(-1, 0, 0); }
  void stepPosZ() { doMove(0, 0, 1); }
  void stepNegZ() { doMove(0, 0, -1); }
  void stepPosY() { doMove(0, 1, 0); }
  void stepNegY() { doMove(0, -1, 0); }

  // Take a one-block step in the direction the agent is currently facing, and
  // sleep for 250ms to limit the agent to the same average walking speed as a
  // human player.
  //
  // Access the agent's current direction as getPlayer().look.yaw
  void stepForward() { client_.discreteStepForward(); }

  // Modify the agent's yaw direction by angle degrees
  void turnAngle(float angle) { client_.turnAngle(angle); }

  // Modify the agent's yaw direction by 90 degrees, left or right
  void turnLeft() { client_.turnLeft(); }
  void turnRight() { client_.turnRight(); }

  // Set the agent's yaw/pitch
  void setLook(float yaw, float pitch) { client_.setLook(Look{yaw, pitch}); }

  // Set the agent's yaw/pitch to look towards the center of a block at xyz
  void lookAt(int x, int y, int z);

  // Send a packet to Place the currently held block in the space directly in
  // front of the agent.
  //
  // Use setHeldItem() to change the block to be placed.
  bool placeBlock(int x, int y, int z) { return client_.placeBlock({x, y, z}); }

  // Send a packet to attack or right-click another entity (a player, minecart, etc).
  bool useEntity(int x, int y, int z) { return client_.useEntity({x, y, z}); }

  // Send a packet to use the currently held item
  //
  // Use setHeldItem() to change the item to be used.
  bool useItem() { return client_.useItem(); }

  // Send a packet to use the currently held item on a target block
  //
  // Use setHeldItem() to change the item to be used.
  bool useItemOnBlock(int x, int y, int z) { return client_.useItemOnBlock({x, y, z}); }

  // Craft an item specified by id/meta
  //
  // On success, returns the number of such items crafted
  // On failure, returns -1
  int craft(py::object arg);

 private:
  void doMove(int x, int y, int z);

  // Fields
  Client client_;
  long chatHistoryIdx_ = 0;  // Index of next chat message to return to python code
};

////////////////
// Exposed to python
////////////////

py::array_t<uint8_t> Agent::getBlocks(int xa, int xb, int ya, int yb, int za, int zb) {
  vector<Block> v;
  int xw = (xb - xa + 1);
  int yw = (yb - ya + 1);
  int zw = (zb - za + 1);
  v.resize(xw * yw * zw);
  client_.getCuboid(v, xa, xb, ya, yb, za, zb);
  return py::array(py::buffer_info(v.data(), sizeof(uint8_t), "b", 4, {yw, zw, xw, 2},
                                   {zw * xw * 2, xw * 2, 2, 1}));
}

py::array_t<uint8_t> Agent::getLocalBlocks(int r) {
  BlockPos p = client_.getPosition().toBlockPos();
  return getBlocks(p.x - r, p.x + r, p.y - r, p.y + r, p.z - r, p.z + r);
}

vector<string> Agent::getIncomingChats() {
  vector<string> incomingChats;
  const vector<string>& chatHistory = client_.getChatHistory();
  string prefix = "<" + client_.getName() + ">";
  for (unsigned int i = chatHistoryIdx_; i < chatHistory.size(); i++) {
    string chat = chatHistory[i];
    if (chat.compare(0, prefix.length(), prefix) != 0) {
      // if not chat.startswith(prefix)
      // i.e. ignore my own chats
      incomingChats.push_back(chat);
    }
  }
  chatHistoryIdx_ = chatHistory.size();
  return incomingChats;
}

bool Agent::setHeldItem(py::object arg) {
  Item item;
  try {
    // Handle (id, meta)
    auto slotPair = arg.cast<pair<uint16_t, uint8_t>>();
    item.id = slotPair.first;
    item.meta = slotPair.second;
  } catch (py::cast_error const&) {
    // Handle id
    item.id = arg.cast<uint16_t>();
    item.meta = 0;
  }
  return client_.setHeldItem(item);
}

vector<Mob> Agent::getMobs() { return client_.getMobs(); }

vector<ItemStack> Agent::getItemStacks() { return client_.getItemStacks(); };

bool Agent::isItemStackOnGround(uint64_t entityId) { return client_.isItemStackOnGround(entityId); }

uint64_t Agent::getInventoryItemCount(uint16_t id, uint8_t meta) {
  return client_.getInventoryItemCount(id, meta);
}

py::tuple Agent::getVision() {
  vector<Block> blocks;
  vector<float> depth;
  client_.getVision(blocks, depth, VISION_H, VISION_W, VISION_MAX_DEPTH);

  auto block_npy = py::array(py::buffer_info(blocks.data(), sizeof(uint8_t), "b", 3,
                                             {VISION_H, VISION_W, 2}, {VISION_W * 2, 2, 1}));
  auto depth_npy = py::array(
      py::buffer_info(depth.data(), sizeof(float), py::format_descriptor<float>::format(), 2,
                      {VISION_H, VISION_W}, {VISION_W * sizeof(float), sizeof(float)}));

  return py::make_tuple(block_npy, depth_npy);
}

optional<BlockPos> Agent::getLineOfSight() {
  Block block;
  float depth;
  BlockPos blockPos;

  bool valid = client_.getLineOfSight(&block, &depth, &blockPos, VISION_MAX_DEPTH);
  if (valid) {
    return blockPos;
  } else {
    return nullopt;
  }
}

optional<BlockPos> Agent::getPlayerLineOfSight(Player player) {
  Block block;
  float depth;
  BlockPos blockPos;

  bool valid = client_.getPlayerLineOfSight(&block, &depth, &blockPos, player, VISION_MAX_DEPTH);
  if (valid) {
    return blockPos;
  } else {
    return nullopt;
  }
}

void Agent::lookAt(int x, int y, int z) {
  Pos eye = client_.getPosition() + Client::HEIGHT;
  Pos dest = BlockPos{x, y, z}.center();
  Pos d = dest - eye;

  float r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
  float yaw = -atan2(d.x, d.z) / 3.14159 * 180;
  float pitch = -asin(d.y / r) / 3.14159 * 180;
  client_.setLook(Look{yaw, pitch});
}

int Agent::craft(py::object arg) {
  try {
    // Handle (id, meta)
    auto p = arg.cast<pair<uint16_t, uint8_t>>();
    return client_.craft(p.first, p.second);
  } catch (py::cast_error const&) {
    // Handle id
    return client_.craft(arg.cast<uint16_t>(), 0);
  }
}

vector<py::tuple> Agent::getChangedBlocks() {
  vector<py::tuple> r;
  vector<BlockWithPos> v = client_.getChangedBlocks();
  r.reserve(v.size());
  for (size_t i = 0; i < v.size(); i++) {
    r.push_back(py::make_tuple(py::make_tuple(v[i].pos.x, v[i].pos.y, v[i].pos.z),
                               py::make_tuple(v[i].block.id, v[i].block.meta)));
  }
  return r;
}

////////////////
// Private
////////////////

void Agent::doMove(int x, int y, int z) {
  if (client_.isCreativeMode()) {
    client_.discreteFly(x, y, z);
  } else if (y != 0) {
    LOG(FATAL) << "Move in y-axis illegal in survival mode";
  } else {
    client_.discreteStep(x, z);
  }
}

////////////////
// Pybind

PYBIND11_MODULE(mc_agent, m) {
  py::class_<Agent>(m, "Agent")
      .def(py::init<const string&, int, const string&>())
      .def("disconnect", &Agent::disconnect, "Disconnect the agent from the server")
      .def("dig", &Agent::dig)
      .def("drop_item_stack_in_hand", &Agent::dropItemStackInHand)
      .def("drop_item_in_hand", &Agent::dropItemInHand)
      .def("drop_inventory_item_stack", &Agent::dropInventoryItemStack)
      .def("set_inventory_slot", &Agent::setInventorySlot)
      .def("get_player_inventory", &Agent::getPlayerInventory)
      .def("get_inventory_item_count", &Agent::getInventoryItemCount)
      .def("get_inventory_items_counts", &Agent::getInventoryItemsCounts)
      .def("send_chat", &Agent::sendChat)
      .def("set_held_item", &Agent::setHeldItem)
      .def("step_pos_x", &Agent::stepPosX)
      .def("step_neg_x", &Agent::stepNegX)
      .def("step_pos_z", &Agent::stepPosZ)
      .def("step_neg_z", &Agent::stepNegZ)
      .def("step_pos_y", &Agent::stepPosY)
      .def("step_neg_y", &Agent::stepNegY)
      .def("step_forward", &Agent::stepForward)
      .def("look_at", &Agent::lookAt)
      .def("set_look", &Agent::setLook)
      .def("turn_angle", &Agent::turnAngle)
      .def("turn_left", &Agent::turnLeft)
      .def("turn_right", &Agent::turnRight)
      .def("place_block", &Agent::placeBlock)
      .def("use_entity", &Agent::useEntity)
      .def("use_item", &Agent::useItem)
      .def("use_item_on_block", &Agent::useItemOnBlock)
      .def("get_item_stacks", &Agent::getItemStacks)
      .def("get_item_stack", &Agent::getItemStack)
      .def("is_item_stack_on_ground", &Agent::isItemStackOnGround)
      .def("craft", &Agent::craft)
      .def("get_blocks", &Agent::getBlocks)
      .def("get_local_blocks", &Agent::getLocalBlocks)
      .def("get_incoming_chats", &Agent::getIncomingChats)
      .def("get_player", &Agent::getPlayer)
      .def("get_mobs", &Agent::getMobs)
      .def("get_other_players", &Agent::getOtherPlayers)
      .def("get_other_player_by_name", &Agent::getOtherPlayerByName)
      .def("get_vision", &Agent::getVision)
      .def("get_line_of_sight", &Agent::getLineOfSight)
      .def("get_player_line_of_sight", &Agent::getPlayerLineOfSight)
      .def("get_changed_blocks", &Agent::getChangedBlocks)
      .def("get_world_age", &Agent::getWorldAge)
      .def("get_time_of_day", &Agent::getTimeOfDay);

  py::class_<BlockPos>(m, "BlockPos")
      .def_readonly("x", &BlockPos::x)
      .def_readonly("y", &BlockPos::y)
      .def_readonly("z", &BlockPos::z)
      .def("__repr__", [](const BlockPos& p) {
        return "<BlockPos (" + to_string(p.x) + ", " + to_string(p.y) + ", " + to_string(p.z) +
               ")>";
      });

  py::class_<Pos>(m, "Pos")
      .def_readonly("x", &Pos::x)
      .def_readonly("y", &Pos::y)
      .def_readonly("z", &Pos::z)
      .def("__repr__", [](const Pos& p) {
        return "<Pos (" + to_string(p.x) + ", " + to_string(p.y) + ", " + to_string(p.z) + ")>";
      });

  py::class_<Look>(m, "Look")
      .def_readonly("yaw", &Look::yaw)
      .def_readonly("pitch", &Look::pitch)
      .def("__repr__", [](const Look& look) {
        return "<Look (" + to_string(look.yaw) + ", " + to_string(look.pitch) + ")>";
      });

  py::class_<Player>(m, "Player")
      .def_readonly("uuid", &Player::uuid)
      .def_readonly("entityId", &Player::entityId)
      .def_readonly("health", &Player::health)
      .def_readonly("foodLevel", &Player::foodLevel)
      .def_readonly("name", &Player::name)
      .def_readonly("pos", &Player::pos)
      .def_readonly("look", &Player::look)
      .def_readonly("mainHand", &Player::mainHand)
      .def("__repr__", [](const Player& p) { return "<Player " + p.name + ">"; });

  py::class_<Mob>(m, "Mob")
      .def_readonly("entityId", &Mob::entityId)
      .def_readonly("mobType", &Mob::mobType)
      .def_readonly("pos", &Mob::pos)
      .def_readonly("look", &Mob::look);

  py::class_<Item>(m, "Item").def_readonly("id", &Item::id).def_readonly("meta", &Item::meta);

  py::class_<ItemStack>(m, "ItemStack")
      .def_readonly("uuid", &ItemStack::uuid)
      .def_readonly("entityId", &ItemStack::entityId)
      .def_readonly("item", &ItemStack::item)
      .def_readonly("pos", &ItemStack::pos);

  py::class_<Slot>(m, "Slot")
      .def_readonly("id", &Slot::id)
      .def_readonly("meta", &Slot::meta)
      .def_readonly("count", &Slot::count)
      .def_readonly("damage", &Slot::damage)
      .def("__repr__", [](const Slot& s) {
        if (s.empty()) {
          return (string) "<Slot/>";
        }
        return "<Slot " + to_string(s.id) + (s.meta ? ":" + to_string(s.meta) : "") + ", " +
               to_string(s.count) + ">";
      });
}
