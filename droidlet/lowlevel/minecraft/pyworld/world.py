"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import time
import numpy as np
from typing import Sequence, Dict
from droidlet.base_util import Pos, Look
from droidlet.lowlevel.minecraft.mc_util import XYZ, IDM
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Slot, Item, ItemStack
from droidlet.shared_data_struct.rotation import look_vec
from droidlet.lowlevel.minecraft.pyworld.fake_mobs import make_mob_opts, MOB_META, SimpleMob
from droidlet.lowlevel.minecraft.pyworld.utils import (
    build_ground,
    make_pose,
    build_coord_shifts,
    shift_coords,
    BEDROCK,
    AIR,
)
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import PASSABLE_BLOCKS


class World:
    def __init__(self, opts, spec):
        self.is_server = False
        self.opts = opts
        self.count = 0
        # sidelength of the cubical npy array defining the extent of the world
        self.sl = opts.sl

        # to be subtracted from incoming coordinates and added to outgoing
        self.coord_shift = spec.get("coord_shift", (0, 0, 0))
        to_npy_coords, from_npy_coords = build_coord_shifts(self.coord_shift)
        self.to_npy_coords = to_npy_coords
        self.from_npy_coords = from_npy_coords

        # TODO point to the actual object?  for now this just stores the eid to avoid collisions
        self.all_eids = {}

        self.blocks = np.zeros((opts.sl, opts.sl, opts.sl, 2), dtype="int32")
        if spec.get("ground_generator"):
            ground_args = spec.get("ground_args", None)
            if ground_args is None:
                spec["ground_generator"](self)
            else:
                spec["ground_generator"](self, **ground_args)
        else:
            build_ground(self)
        # TODO make some machinery to never allow these to go out of sync- adding blocks
        # and removing them should go through interface
        nz_blocks = [
            (int(l[0]), int(l[1]), int(l[2]))
            for l in zip(*np.nonzero(self.blocks[:, :, :, 0] > 0))
        ]
        self.blocks_dict = {l: tuple(int(i) for i in self.blocks[l].tolist()) for l in nz_blocks}

        self.mobs = []
        for m in spec["mobs"]:
            m.add_to_world(self)
        self.items = {}
        for i in spec.get("items"):
            i.add_to_world(self)
        self.players = {}
        # TODO make this more robust
        # rn it is a dict for each player entityId with
        # item typeNames as keys, and a list of item objects as values
        self.player_inventories = {}
        for p in spec["players"]:
            # FIXME! world should assign and manage entityId
            if hasattr(p, "add_to_world"):
                p.add_to_world(self)
            else:
                self.players[p.entityId] = p
                # players cannot have anything in their inventory at init, TODO
                self.player_inventories[p.entityId] = {}

        self.agent_data = spec["agent"]
        self.chat_log = []

        # keep a list of blocks changed since the last call of
        # get_changed_blocks of each agent
        # TODO more efficient?
        self.changed_blocks_store = {}

        # keep a list of chats since the last call of
        # get_incoming_chats of each agent
        self.incoming_chats_store = {}

        # FIXME
        self.mob_opt_maker = make_mob_opts
        self.mob_maker = SimpleMob

        # TODO: Add item stack maker?

        if hasattr(opts, "world_server") and opts.world_server:
            port = getattr(opts, "port", 25565)
            self.server = self.setup_server(port=port)

    def new_eid(self, entityId=None):
        count = 0
        while entityId is None:
            eid = int(np.random.randint(0, 1000000000))
            if self.all_eids.get(eid, False):
                count = count + 1
            else:
                entityId = eid
            if count > 10000:
                raise Exception(
                    "Tried to add a new entity 10000 times, could not make an entityId"
                )
        if type(entityId) is not int:
            raise Exception(
                "tried to add a new entity with bad entityId type {}".format(type(entityId))
            )
        if entityId < 1:
            raise Exception(
                "tried to add a new entity with negative entityId  {}".format(entityId)
            )
        self.all_eids[entityId] = True
        return entityId

    def set_count(self, count):
        self.count = count

    def step(self):
        for m in self.mobs:
            m.step()
            for item_eid in getattr(m, "inventory", []):
                item = self.items[item_eid]
                item.update_position(*m.pos)

        for eid, p in self.players.items():
            if hasattr(p, "step") and not getattr(p, "no_step_from_world", False):
                p.step()

        for eid, item in self.items.items():
            if self.players.get(item.holder_entityId):
                item.update_position(*self.players[item.holder_entityId].pos)

        self.count += 1

        if self.is_server:
            self.broadcast_updates()

    def broadcast_updates(self):
        # broadcast updates
        players = [
            {
                "name": player.name,
                "x": player.pos.x,
                "y": player.pos.y,
                "z": player.pos.z,
                "yaw": player.look.yaw,
                "pitch": player.look.pitch,
            }
            for player in self.get_players()
            if player.name in ["craftassist_agent", "dashboard"]
        ]
        mobs = [
            {
                "entityId": m.entityId,
                "pos": m.pos,
                "look": m.look,
                "mobType": m.mobType,
                "color": m.color,
                "name": m.mobname,
            }
            for m in self.mobs
        ]
        items = self.get_items()
        # FIXME !!!! item_stacks
        payload = {
            "status": "updateVoxelWorldState",
            "world_state": {"agent": players, "mob": mobs, "item_stack": items},
            "backend": "pyworld",
        }
        # print(f"Server stepping, payload: {payload}")
        self.server.emit("updateVoxelWorldState", payload)

    def broadcast_block_update(self, loc, idm):
        blocks = [((int(loc[0]), int(loc[1]), int(loc[2])), (int(idm[0]), int(idm[1])))]
        payload = {
            "status": "updateVoxelWorldState",
            "world_state": {"block": blocks, "backend": "pyworld"},
        }
        self.server.emit("updateVoxelWorldState", payload)

    def get_height_map(self):
        """
        get the ground height at each location, to maybe place items, mobs, and players
        """
        height_map = np.zeros((self.sl, self.sl))
        for l, idm in self.blocks_dict.items():
            if l[1] > height_map[l[0], l[2]] and idm[0] > 0:
                height_map[l[0], l[2]] = l[1]
        return height_map

    def place_block(self, block, force=False):
        loc, idm = block
        if idm[0] == 383:
            # its a mob...
            try:
                # FIXME handle unknown mobs/mobs not in list
                m = SimpleMob(make_mob_opts(MOB_META[idm[1]]), start_pos=loc)
                m.agent_built = True
                m.add_to_world(self)
                return True
            except:
                return False
        # mobs keep loc in real coords, block objects we convert to the numpy index
        loc = tuple(int(i) for i in self.to_npy_coords(loc))
        idm = tuple(int(s) for s in idm)
        try:  # FIXME only allow placing non-air blocks in air locations?
            if tuple(int(i) for i in self.blocks[loc]) != BEDROCK or force:
                self.blocks[loc] = idm
                if self.is_server:
                    self.broadcast_block_update(loc, idm)
                for sid, store in self.changed_blocks_store.items():
                    store[tuple(loc)] = idm
                if idm != AIR:
                    self.blocks_dict[loc] = idm
                else:
                    self.blocks_dict.pop(loc, None)
                return True
            else:
                return False
        except:
            # FIXME this will return False if the block was placed but not stored in the changed blocks store
            return False

    def dig(self, loc: XYZ):
        return self.place_block((loc, AIR))

    def blocks_to_dict(self):
        d = {}
        nz = np.transpose(self.blocks[:, :, :, 0].nonzero())
        for loc in nz:
            l = tuple(loc.tolist())
            d[self.from_npy_coords(l)] = tuple(self.blocks[l[0], l[1], l[2], :])
        return d

    def get_idm_at_locs(self, xyzs: Sequence[XYZ]) -> Dict[XYZ, IDM]:
        """Return the ground truth block state"""
        d = {}
        for (x, y, z) in xyzs:
            B = self.get_blocks(x, x, y, y, z, z)
            d[(x, y, z)] = tuple(B[0, 0, 0, :])
        return d

    def get_mobs(self):
        return [m.get_info() for m in self.mobs]

    def get_items(self):
        return [i.get_info() for i in self.items.values()]

    def get_item_stacks(self):
        item_stacks = []
        for item in self.get_items():
            pos = Pos(item["x"], item["y"], item["z"])
            item_stacks.append(
                ItemStack(
                    Slot(item["id"], item["meta"], 1), pos, item["entityId"], item["typeName"]
                )
            )
        return item_stacks

    def get_player_info(self, eid):
        """
        returns a Player struct
        or None if eid is not a player
        it is assumed that if p implements its own get_info() method, that
        returns a Player struct
        """
        p = self.players.get(eid)
        if not p:
            return
        if hasattr(p, "get_info"):
            return p.get_info()
        else:
            return p

    def get_players(self):
        return [self.get_player_info(eid) for eid in self.players]

    def get_blocks(self, xa, xb, ya, yb, za, zb, transpose=True):
        xa, ya, za = self.to_npy_coords((xa, ya, za))
        xb, yb, zb = self.to_npy_coords((xb, yb, zb))
        M = np.array((xb, yb, zb))
        m = np.array((xa, ya, za))
        szs = M - m + 1
        B = np.zeros((szs[0], szs[1], szs[2], 2), dtype="uint8")
        B[:, :, :, 0] = 7
        xs, ys, zs = [0, 0, 0]
        xS, yS, zS = szs
        if xb < 0 or yb < 0 or zb < 0 or xa > self.sl - 1 or ya > self.sl - 1 or za > self.sl - 1:
            if transpose:
                B = B.transpose(1, 2, 0, 3)
            return B
        if xb > self.sl - 1:
            xS -= xb - (self.sl - 1)
            xb = self.sl - 1
        if yb > self.sl - 1:
            yS -= yb - (self.sl - 1)
            yb = self.sl - 1
        if zb > self.sl - 1:
            zS -= zb - (self.sl - 1)
            zb = self.sl - 1
        if xa < 0:
            xs = -xa
            xa = 0
        if ya < 0:
            ys = -ya
            ya = 0
        if za < 0:
            zs = -za
            za = 0

        B[xs:xS, ys:yS, zs:zS, :] = self.blocks[xa : xb + 1, ya : yb + 1, za : zb + 1, :]
        if transpose:
            B = B.transpose(1, 2, 0, 3)
        return B

    def get_line_of_sight(self, pos, yaw, pitch, loose=0):
        # it is assumed lv is unit normalized
        pos = tuple(self.to_npy_coords(pos))
        lv = look_vec(yaw, pitch)
        dt = 1.0
        for n in range(2 * self.sl):
            p = tuple(np.round(np.add(pos, n * dt * lv)).astype("int32"))
            for i in range(-loose, loose + 1):
                for j in range(-loose, loose + 1):
                    for k in range(-loose, loose + 1):
                        sp = tuple(np.add(p, (i, j, k)))
                        if all([x >= 0 for x in sp]) and all([x < self.sl for x in sp]):
                            if tuple(self.blocks[sp]) != AIR:
                                # TODO: deal with close blocks artifacts,
                                # etc
                                pos = self.from_npy_coords(sp)
                                return tuple(int(l) for l in pos)
        return

    def add_incoming_chat(self, chat: str, speaker_name: str):
        """Add a chat to memory as if it was just spoken by SPEAKER"""
        self.chat_log.append("<" + speaker_name + ">" + " " + chat)

    def connect_player(self, sid, data):
        # FIXME, this probably won't actually work
        if self.connected_sids.get(sid) is not None:
            print("reconnecting eid {} (sid {})".format(self.connected_sids["sid"], sid))
            return
        for player_eid in self.connected_sids.values():
            player_info = self.get_player_info(player_eid)
            if player_info.name == data.get("name"):
                print("reconnecting eid {} (sid {})".format(player_eid, sid))
                return

        x, y, z, pitch, yaw = make_pose(
            self.sl,
            self.sl,
            loc=data.get("loc"),
            pitchyaw=data.get("pitchyaw"),
            height_map=self.get_height_map(),
        )
        entityId = self.new_eid(entityId=data.get("entityId"))
        # FIXME
        name = data.get("name", "anonymous")
        p = Player(entityId, name, Pos(int(x), int(y), int(z)), Look(float(yaw), float(pitch)), Item(None, None))
        self.players[entityId] = p
        self.connected_sids[sid] = entityId

        if data.get("player_type") == "agent":
            self.changed_blocks_store[sid] = {}
            self.incoming_chats_store[sid] = []

    def get_player_by_name(self, player_name):
        for player_eid in self.connected_sids.values():
            player_info = self.get_player_info(player_eid)
            if player_info.name == player_name:
                return {"player": player_info}
        return None

    def player_pick_drop_items(self, player_eid, data, action="pick"):
        """
        data should be a list of entityIds
        """
        count = 0
        for eid in data:
            item = self.items.get(eid)
            if item is not None:
                # TODO inform caller if eid doesn't exist?
                if action == "pick":
                    # TODO check it isn't already attached?
                    item.attach_to_entity(player_eid)
                    count += 1
                else:
                    if item.holder_entityId == player_eid:
                        item.attach_to_entity(-1)
                        count += 1
        return count

    def check_in_bounds(self, player, pos):
        if player.name == "dashboard":
            lowerb = (self.sl / 3, 0, self.sl / 3)
            upperb = (2 * self.sl / 3, self.sl / 3 - 1, 2 * self.sl / 3)
        else:
            lowerb = (0, 0, 0)
            upperb = (self.sl, self.sl - 1, self.sl)
        if (
            pos[0] >= lowerb[0]
            and pos[1] >= lowerb[1]
            and pos[2] >= lowerb[2]
            and pos[0] < upperb[0]
            and pos[1] < upperb[1]
            and pos[2] < upperb[2]
        ):
            return True
        else:
            return False

    def setup_server(self, port=25565):
        import socketio
        import eventlet

        self.is_server = True

        server = socketio.Server(async_mode="eventlet", cors_allowed_origins="*")
        self.server = server
        self.connected_sids = {}

        self.start_time = time.time()
        self.get_time = lambda: time.time() - self.start_time

        @server.event
        def connect(sid, environ):
            print("connect ", sid)

        @server.event
        def disconnect(sid):
            print("disconnect ", sid)

        # the player init is separate bc connect special format, FIXME?
        @server.on("init_player")
        def init_player_event(sid, data):
            self.connect_player(sid, data)

        @server.on("getVoxelWorldInitialState")
        def get_voxel_world_initial_state(sid):
            print("test get VW initial status")
            blocks = self.blocks_to_dict()
            blocks = [
                ((int(xyz[0]), int(xyz[1]), int(xyz[2])), (int(idm[0]), int(idm[1])))
                for xyz, idm in blocks.items()
            ]

            payload = {
                "status": "updateVoxelWorldState",
                "world_state": {"block": blocks},
                "backend": "pyworld",
            }
            # print(f"Initial payload: {payload}")
            server.emit("updateVoxelWorldState", payload)

        @server.on("get_world_info")
        def get_world_info(sid):
            print("get world info")
            return {"sl": self.sl, "coord_shift": self.coord_shift}

        @server.on("send_chat")
        def broadcast_chat(sid, chat_text):
            eid = self.connected_sids.get(sid)
            player_struct = self.get_player_info(eid)
            chat_with_name = "<{}> {}".format(player_struct.name, chat_text)
            for other_sid, store in self.incoming_chats_store.items():
                if sid != other_sid:
                    store.append(chat_with_name)

        @server.on("get_incoming_chats")
        def get_chats(sid):
            chats = {"chats": self.incoming_chats_store[sid]}
            self.incoming_chats_store[sid] = []
            return chats

        @server.on("line_of_sight")
        def los_event(sid, data):
            if data.get("pos"):
                pos = self.get_line_of_sight(data["pos"], data["yaw"], data["pitch"])
            else:
                if data.get("entityId"):
                    eid = data["entityId"]
                else:
                    eid = self.connected_sids.get(sid)
                if not eid:
                    raise Exception(
                        "player connected, asking for line of sight, but sid does not match any entityId"
                    )
                player_struct = self.get_player_info(eid)
                pos = self.get_line_of_sight(player_struct.pos, *player_struct.look)
            pos = pos or ""
            return {"pos": pos}

        # warning: it is assumed that if a player is using the sio event to set look,
        # the only thing stored here is a Player struct, not some more complicated object

        @server.on("set_look")
        def set_agent_look(sid, data):
            eid = self.connected_sids.get(sid)
            player_struct = self.get_player_info(eid)
            if not player_struct:
                # FIXME only works for dashboard reconnect
                for player_eid in self.connected_sids.values():
                    player_info = self.get_player_info(player_eid)
                    if player_info.name == "dashboard":
                        player_struct = self.get_player_info(player_eid)
                        eid = player_eid
                        break

            new_look = Look(data["yaw"], data["pitch"])
            self.players[eid] = self.players[eid]._replace(look=new_look)

        @server.on("rel_move")
        def move_agent_rel(sid, data):
            eid = self.connected_sids.get(sid)
            player_struct = self.get_player_info(eid)
            x, y, z = player_struct.pos
            if data.get("forward"):
                pass  # FIXME!!!!
            else:
                x += data.get("x", 0)
                y += data.get("y", 0)
                z += data.get("z", 0)
            nx, ny, nz = self.to_npy_coords((x, y, z))
            if self.check_in_bounds(player_struct, (nx, ny, nz)):
                if (
                    self.blocks[nx, ny, nz, 0] in PASSABLE_BLOCKS
                    and self.blocks[nx, ny + 1, nz, 0] in PASSABLE_BLOCKS
                ):
                    new_pos = Pos(x, y, z)
                    self.players[eid] = self.players[eid]._replace(pos=new_pos)
                else:
                    print(f"{player_struct.name} tried to move somewhere impossible")
            else:
                print(f"{player_struct.name} tried to move somewhere impossible")

        @server.on("abs_move")
        def move_agent_abs(sid, data):
            eid = self.connected_sids.get(sid)
            player_struct = self.get_player_info(eid)
            # FIXME sid lost on page refresh, hacky workaround
            if not player_struct:
                for player_eid in self.connected_sids.values():
                    player_info = self.get_player_info(player_eid)
                    if player_info.name == "dashboard":
                        player_struct = self.get_player_info(player_eid)
                        eid = player_eid
                        break

            x, y, z = player_struct.pos
            x = data.get("x", 0)
            y = data.get("y", 0)
            z = data.get("z", 0)

            nx, ny, nz = self.to_npy_coords((x, y, z))
            if self.check_in_bounds(player_struct, (nx, ny, nz)):
                if (
                    self.blocks[int(nx), int(ny), int(nz), 0] in PASSABLE_BLOCKS
                    and self.blocks[int(nx), int(ny) + 1, int(nz), 0] in PASSABLE_BLOCKS
                ):
                    new_pos = Pos(x, y, z)
                    self.players[eid] = self.players[eid]._replace(pos=new_pos)
                else:
                    print(f"{player_struct.name} tried to move somewhere impossible")
                    print(player_struct.pos)
            else:
                print(f"{player_struct.name} tried to move somewhere impossible")
                print(player_struct.pos)

        @server.on("set_held_item")
        def set_agent_mainhand(sid, data):
            if data.get("idm") is not None:
                eid = self.connected_sids.get(sid)
                player_struct = self.get_player_info(eid)
                new_item = Item(*data["idm"])
                self.players[eid] = self.players[eid]._replace(mainHand=new_item)

        @server.on("place_block")
        def place_mainhand(sid, data):
            eid = self.connected_sids.get(sid)
            player_struct = self.get_player_info(eid)
            idm = player_struct.mainHand
            if data.get("loc"):
                placed = self.place_block((data["loc"], idm))
                return placed

        @server.on("dig")
        def agent_dig(sid, data):
            if data.get("loc"):
                placed = self.dig(data["loc"])
                return placed

        @server.on("get_player")
        def get_player_by_sid(sid):
            eid = self.connected_sids.get(sid)
            return {"player": self.get_player_info(eid)}

        @server.on("get_players")
        def get_all_players(sid):
            return self.get_players()

        @server.on("get_mobs")
        def send_mobs(sid):
            mobs = self.get_mobs()
            serialized_mobs = []
            for m in mobs:
                x, y, z = m.pos
                yaw, pitch = m.look
                serialized_mobs.append((m.entityId, m.mobType, x, y, z, yaw, pitch))
            return serialized_mobs

        @server.on("get_item_info")
        def send_items(sid):
            return self.get_items()

        @server.on("pick_items")
        def pick_items(sid, data):
            """
            data should be a list of entityId
            """
            player_eid = self.connected_sids[sid]
            return self.player_pick_drop_items(player_eid, data, action="pick")

        @server.on("drop_items")
        def drop_items(sid, data):
            """
            data should be a list of entityId
            """
            player_eid = self.connected_sids[sid]
            return self.player_pick_drop_items(player_eid, data, action="drop")

        @server.on("get_changed_blocks")
        def changed_blocks(sid):
            eid = self.connected_sids.get(sid)
            blocks = self.changed_blocks_store[sid]
            # can't send dicts with tuples for keys :(
            blocks = {str(k): v for k, v in blocks.items()}
            self.changed_blocks_store[sid] = {}
            return blocks

        @server.on("get_blocks")
        def get_blocks_dict(sid, data):
            x, X, y, Y, z, Z = data["bounds"]
            npy = self.get_blocks(x, X, y, Y, z, Z, transpose=False)
            nz_locs = list(zip(*np.nonzero(npy[:, :, :, 0])))
            nz_idm_locs = [
                (int(l[0]) + int(x), int(l[1]) + int(y), int(l[2]) + int(z)) for l in nz_locs
            ]
            nz_idms = []
            for l in nz_idm_locs:
                if (
                    l[0] >= 0
                    and l[0] < self.sl
                    and l[1] >= 0
                    and l[1] < self.sl
                    and l[2] >= 0
                    and l[2] < self.sl
                ):
                    nz_idms.append(tuple(int(i) for i in self.blocks[l]))
                else:
                    nz_idms.append((7, 0))
            nz_locs = [(int(x), int(y), int(z)) for x, y, z in nz_locs]
            flattened_blocks = [nz_locs[i] + nz_idms[i] for i in range(len(nz_locs))]
            return flattened_blocks

        @server.on("start_world")
        def start_world(sid):
            self.start()

        @server.on("step_world")
        def step_world(sid):
            self.step()

        app = socketio.WSGIApp(server)
        eventlet.wsgi.server(eventlet.listen(("", port)), app)
        return server


if __name__ == "__main__":

    class Opt:
        pass

    spec = {"players": [], "mobs": [], "items": [], "coord_shift": (0, 0, 0), "agent": {}}
    world_opts = Opt()
    world_opts.sl = 16 * 3
    world_opts.world_server = True
    world_opts.port = 6002
    world = World(world_opts, spec)
