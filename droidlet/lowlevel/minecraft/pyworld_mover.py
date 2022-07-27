"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import time
import socketio
import sys
import logging
from droidlet.base_util import Pos, Look
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Slot, Item, ItemStack, Mob
from droidlet.lowlevel.minecraft.pyworld.utils import build_coord_shifts


class DataCallback:
    def __call__(self, data):
        self.data = data


def wait_for_data(data_callback, step=0.002, max_steps=100):
    data = None
    count = 0
    while not data:
        data = getattr(data_callback, "data", None)
        if data is not None or count > max_steps:
            return data
        count += 1
        time.sleep(step)


class PyWorldMover:
    def __init__(self, port=25565, ip="localhost"):
        sio = socketio.Client()
        try:
            sio.connect("http://{}:{}".format(ip, port))
            time.sleep(0.1)
            sio.emit("init_player", {"player_type": "agent", "name": "craftassist_agent"})
        except:
            raise Exception("unable to connect to server on port {} at ip {}".format(port, ip))

        logging.info("connected to server on port {} at ip {}".format(port, ip))

        self.sio = sio
        D = DataCallback()
        self.sio.emit("get_world_info", callback=D)
        info = wait_for_data(D)
        self.sl = info["sl"]
        self.world_coord_shift = info["coord_shift"]
        to_npy_coords, from_npy_coords = build_coord_shifts(self.world_coord_shift)
        self.to_npy_coords = to_npy_coords
        self.from_npy_coords = from_npy_coords
        player_struct = self.get_player()
        self.entityId = player_struct.entityId

    def set_look(self, yaw, pitch):
        self.sio.emit("set_look", {"yaw": float(yaw), "pitch": float(pitch)})

    # FIXME: make a "cuberite" style pick mode, and an explicit pick mode
    def step_pos_x(self):
        self.sio.emit("rel_move", {"x": 1})

    #        self.pick_nearby_items()

    def step_neg_x(self):
        self.sio.emit("rel_move", {"x": -1})

    #       self.pick_nearby_items()

    def step_pos_y(self):
        self.sio.emit("rel_move", {"y": 1})

    #       self.pick_nearby_items()

    def step_neg_y(self):
        self.sio.emit("rel_move", {"y": -1})

    #        self.pick_nearby_items()

    def step_pos_z(self):
        self.sio.emit("rel_move", {"z": 1})

    #        self.pick_nearby_items()

    def step_neg_z(self):
        self.sio.emit("rel_move", {"z": -1})

    #        self.pick_nearby_items()

    def step_forward(self):
        self.sio.emit("rel_move", {"forward": 1})

    #        self.pick_nearby_items()

    def set_held_item(self, idm):
        idm = (int(idm[0]), int(idm[1]))
        self.sio.emit("set_held_item", {"idm": idm})

    def dig(self, x, y, z):
        D = DataCallback()
        self.sio.emit("dig", {"loc": [int(x), int(y), int(z)]}, callback=D)
        # return True if the world says the block was dug, False otherwise
        placed = wait_for_data(D)
        # this is sketchy: if world doesn't respond in time, block placement will
        # show False even if block was placed....  TODO? check with a get_blocks?
        if placed is None:
            placed = False
        return placed

    def place_block(self, x, y, z):
        """place the block in mainhand.  does nothing if mainhand empty"""
        D = DataCallback()
        self.sio.emit("place_block", {"loc": [int(x), int(y), int(z)]}, callback=D)
        # return True if the world says the block was placed, False otherwise
        placed = wait_for_data(D)
        # this is sketchy: if world doesn't respond in time, block placement will
        # show False even if block was placed....  TODO? check with a get_blocks?
        if placed is None:
            placed = False
        return placed

    def get_player(self):
        D = DataCallback()
        self.sio.emit("get_player", callback=D)
        try:
            entityId, name, pos, look, mainhand, _ = wait_for_data(D)["player"]
            if mainhand is not None:
                mainhand = Item(*mainhand)
                return Player(entityId, name, Pos(*pos), Look(*look), mainhand)
            else:
                return Player(entityId, name, Pos(*pos), Look(*look))
        except:
            return None

    def get_line_of_sight(self):
        D = DataCallback()
        self.sio.emit("line_of_sight", {}, callback=D)
        pos = wait_for_data(D)["pos"]
        if pos != "":
            return Pos(*pos)
        else:
            return None

    def get_player_line_of_sight(self, player_struct):
        D = DataCallback()
        pos = player_struct.pos
        look = player_struct.look
        pose_data = {
            "pos": (float(pos.x), float(pos.y) + 1, float(pos.z)),
            "yaw": float(look.yaw),
            "pitch": float(look.pitch),
        }
        self.sio.emit("line_of_sight", pose_data, callback=D)
        pos = wait_for_data(D)["pos"]
        if pos == "":
            return None
        else:
            return Pos(*pos)

    def get_item_stacks(self, holder_entityId=-1, get_all=False):
        """
        by default
        only return items not in any agent's inventory, matching cuberite
        returns a list of ItemStacks
        """
        D = DataCallback()
        self.sio.emit("get_item_info", callback=D)
        items = wait_for_data(D)
        # TODO "stacks" with count, like MC?
        # right now make a separate "item_stack" for each one
        item_stacks = []
        for item in items:
            if item["holder_entityId"] == holder_entityId or get_all:
                pos = Pos(item["x"], item["y"], item["z"])
                item_stacks.append(
                    [
                        ItemStack(
                            Slot(item["id"], item["meta"], 1),
                            pos,
                            item["entityId"],
                            item["typeName"],
                        ),
                        item["holder_entityId"],
                        item["properties"],
                    ]
                )
        return item_stacks

    def get_inventory_item_count(self, bid, meta, entityId=None):
        item_stacks = self.get_item_stacks(holder_entityId=self.entityId)
        count = 0
        for item_stack in item_stacks:
            correct_idm = item_stack.item.id == bid[0] and item_stack.item.meta == bid[1]
            if item_stack.entityId == entityId or correct_idm:
                count = count + 1
        return count

    # FIXME, make a regular pick, split the Task into a Move and Pick.  rn
    # just following MC blindly
    def pick_nearby_items(self, pick_range=1.5):
        pos = np.array(self.get_player().pos)
        item_stacks = self.get_item_stacks()
        for item_stack in item_stacks:
            if np.linalg.norm(np.array(item_stack.pos) - pos) < pick_range:
                D = DataCallback()
                self.sio.emit("pick_items", [int(item_stack.entityId)], callback=D)
                count = wait_for_data(D)
                return count

    def pick_entityId(self, eid):
        item_stacks = self.get_item_stacks(get_all=True)
        matching_items = [i for i in item_stacks if i[0].entityId == eid]
        if len(matching_items) >= 1:
            logging.info(f"Picking up item: {matching_items[0][0]}")
            D = DataCallback()
            self.sio.emit("pick_items", [int(matching_items[0][0].entityId)], callback=D)
            count = wait_for_data(D)
            return count
        else:
            logging.error("Tried to pick up an item that doesn't exist in the world")
            return 0

    def drop_inventory_item_stack(self, bid=None, meta=None, count=1, entityId=-1):
        dropped = 0
        item_stacks = self.get_item_stacks(holder_entityId=self.entityId)
        assert bid is not None or entityId > 0
        for item_stack in item_stacks:
            correct_idm = bid is None or (
                item_stack.item.id == bid[0] and item_stack.item.meta == bid[1]
            )
            if item_stack.entityId == entityId or correct_idm:
                self.sio.emit("drop_items", [int(item_stack.entityId)])
                dropped = dropped + 1
                if dropped > count:
                    return

    def get_changed_blocks(self):
        D = DataCallback()
        self.sio.emit("get_changed_blocks", callback=D)
        blocks = wait_for_data(D)

        def decode_str_loc(x):
            return tuple(int(z) for z in x.strip("(").strip(")").split(","))

        if blocks:
            # can't send dicts with tuples for keys :(
            blocks = {decode_str_loc(loc): idm for loc, idm in blocks.items()}
        return blocks

    def get_blocks(self, x, X, y, Y, z, Z):
        """
        returns an (Y-y+1) x (Z-z+1) x (X-x+1) x 2 numpy array B of the blocks
        in the rectanguloid with bounded by the input coordinates (including endpoints).
        Input coordinates are in droidlet coordinates; and the output array is
        in yzxb permutation, where B[0,0,0,:] corresponds to the id and meta of
        the block at x, y, z

        TODO we don't need yzx orientation anymore...
        """
        D = DataCallback()
        self.sio.emit(
            "get_blocks", {"bounds": (int(x), int(X), int(y), int(Y), int(z), int(Z))}, callback=D
        )
        flattened_blocks = wait_for_data(D)
        npy_blocks = np.zeros((Y - y + 1, Z - z + 1, X - x + 1, 2), dtype="int32")
        if flattened_blocks:
            for b in flattened_blocks:
                npy_blocks[b[1], b[2], b[0]] = [b[3], b[4]]
        return npy_blocks

    def send_chat(self, chat_text):
        self.sio.emit("send_chat", chat_text)

    # TODO is this supposed to include the self Player ?
    def get_other_players(self):
        D = DataCallback()
        self.sio.emit("get_players", callback=D)
        players = wait_for_data(D)
        out = []
        for p in players:
            entityId, name, pos, look, mainhand, _ = p
            if entityId != self.entityId:
                if mainhand is not None:
                    mainhand = Item(*mainhand)
                    out.append(Player(entityId, name, Pos(*pos), Look(*look), mainhand))
                else:
                    out.append(Player(entityId, name, Pos(*pos), Look(*look)))
        return out

    def get_incoming_chats(self):
        try:
            D = DataCallback()
            self.sio.emit("get_incoming_chats", callback=D)
            chats = wait_for_data(D)
        except socketio.exceptions.BadNamespaceError:
            logging.info("Exiting on bad SIO namespace")
            sys.exit()

        if chats is not None:
            chats = chats["chats"]
        else:
            chats = []
        return chats

    def get_mobs(self):
        D = DataCallback()
        self.sio.emit("get_mobs", callback=D)
        serialized_mobs = wait_for_data(D)
        mobs = []
        for m in serialized_mobs:
            mobs.append(Mob(m[0], m[1], Pos(m[2], m[3], m[4]), Look(m[5], m[6])))
        return mobs


### NOT DONE:
#    "drop_item_stack_in_hand",
#    "drop_item_in_hand",
#    "drop_inventory_item_stack",
#    "set_inventory_slot",
#    "get_player_inventory",
#    "get_inventory_item_count",
#    "get_inventory_items_counts",
#    "step_forward",
#    "use_entity",
#    "use_item",
#    "use_item_on_block",
#    "craft",
#    "get_world_age",
#    "get_time_of_day",
#    "get_vision",
#    "disconnect",


if __name__ == "__main__":
    m = PyWorldMover(port=6002)
    m.set_held_item((1, 0))
    r = m.place_block(10, 10, 10)
    print(r)
    print(m.get_changed_blocks())
    r = m.dig(10, 10, 10)
    print(r)
    print(m.get_changed_blocks())
