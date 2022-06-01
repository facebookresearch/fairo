"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from typing import Sequence, Dict
from droidlet.base_util import Pos, Look
from droidlet.lowlevel.minecraft.mc_util import XYZ, IDM
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Item
from droidlet.shared_data_struct.rotation import look_vec
from droidlet.lowlevel.minecraft.pyworld.fake_mobs import make_mob_opts, MOB_META, SimpleMob
from droidlet.lowlevel.minecraft.pyworld.utils import (
    build_ground,
    make_pose,
    build_coord_shifts,
    shift_coords,
)
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import PASSABLE_BLOCKS


class World:
    def __init__(self, opts, spec):
        self.opts = opts
        self.count = 0
        # sidelength of the cubical npy array defining the extent of the world
        self.sl = opts.sl

        # to be subtracted from incoming coordinates and added to outgoing
        self.coord_shift = spec.get("coord_shift", (0, 0, 0))
        to_npy_coords, from_npy_coords = build_coord_shifts(self.coord_shift)
        self.to_npy_coords = to_npy_coords
        self.from_npy_coords = from_npy_coords

        self.blocks = np.zeros((opts.sl, opts.sl, opts.sl, 2), dtype="int32")
        if spec.get("ground_generator"):
            ground_args = spec.get("ground_args", None)
            if ground_args is None:
                spec["ground_generator"](self)
            else:
                spec["ground_generator"](self, **ground_args)
        else:
            build_ground(self)
        self.mobs = []
        for m in spec["mobs"]:
            m.add_to_world(self)
        self.item_stacks = []
        for i in spec["item_stacks"]:
            i.add_to_world(self)
        self.players = {}
        for p in spec["players"]:
            if hasattr(p, "add_to_world"):
                p.add_to_world(self)
            else:
                self.players[p.entityId] = p
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
            self.setup_server(port=port)

    def set_count(self, count):
        self.count = count

    def step(self):
        for m in self.mobs:
            m.step()
        for eid, p in self.players.items():
            if hasattr(p, "step"):
                p.step()
        self.count += 1

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
        loc = tuple(self.to_npy_coords(loc))
        idm = tuple(int(s) for s in idm)
        try:  # FIXME only allow placing non-air blocks in air locations?
            if tuple(self.blocks[loc]) != (7, 0) or force:
                self.blocks[loc] = idm
                for sid, store in self.changed_blocks_store.items():
                    store[tuple(loc)] = idm
                return True
            else:
                return False
        except:
            # FIXME this will return False if the block was placed but not stored in the changed blocks store
            return False

    def dig(self, loc: XYZ):
        return self.place_block((loc, (0, 0)))

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

    def get_item_stacks(self):
        return [i.get_info() for i in self.item_stacks]

    def get_blocks(self, xa, xb, ya, yb, za, zb, transpose=True):
        xa, ya, za = self.to_npy_coords((xa, ya, za))
        xb, yb, zb = self.to_npy_coords((xb, yb, zb))
        M = np.array((xb, yb, zb))
        m = np.array((xa, ya, za))
        szs = M - m + 1
        B = np.zeros((szs[1], szs[2], szs[0], 2), dtype="uint8")
        B[:, :, :, 0] = 7
        xs, ys, zs = [0, 0, 0]
        xS, yS, zS = szs
        if xb < 0 or yb < 0 or zb < 0:
            return B
        if xa > self.sl - 1 or ya > self.sl - 1 or za > self.sl - 1:
            return B
        if xb > self.sl - 1:
            xS = self.sl - xa
            xb = self.sl - 1
        if yb > self.sl - 1:
            yS = self.sl - ya
            yb = self.sl - 1
        if zb > self.sl - 1:
            zS = self.sl - za
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
        pre_B = self.blocks[xa : xb + 1, ya : yb + 1, za : zb + 1, :]
        # pre_B = self.blocks[ya : yb + 1, za : zb + 1, xa : xb + 1, :]
        B[ys:yS, zs:zS, xs:xS, :] = pre_B.transpose(1, 2, 0, 3)
        if transpose:
            return B
        else:
            return pre_B

    def get_line_of_sight(self, pos, yaw, pitch):
        # it is assumed lv is unit normalized
        pos = tuple(self.to_npy_coords(pos))
        lv = look_vec(yaw, pitch)
        dt = 1.0
        for n in range(2 * self.sl):
            p = tuple(np.round(np.add(pos, n * dt * lv)).astype("int32"))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        sp = tuple(np.add(p, (i, j, k)))
                        if all([x >= 0 for x in sp]) and all([x < self.sl for x in sp]):
                            if tuple(self.blocks[sp]) != (0, 0):
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
        # FIXME add height map
        x, y, z, pitch, yaw = make_pose(
            self.sl, self.sl, loc=data.get("loc"), pitchyaw=data.get("pitchyaw")
        )
        entityId = data.get("entityId") or int(np.random.randint(0, 100000000))
        # FIXME
        name = data.get("name", "anonymous")
        p = Player(entityId, name, Pos(int(x), int(y), int(z)), Look(float(yaw), float(pitch)))
        self.players[entityId] = p
        self.connected_sids[sid] = entityId

        if data.get("player_type") == "agent":
            self.changed_blocks_store[sid] = {}
            self.incoming_chats_store[sid] = []

    def setup_server(self, port=25565):
        import socketio
        import eventlet
        import time

        server = socketio.Server(async_mode="eventlet", cors_allowed_origins='*')
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
        def testing(sid):
            print('test get VW initial status')

        @server.on("get_world_info")
        def get_world_info(sid):
            print("get world info")
            return {"sl": self.sl, "coord_shift": self.coord_shift}

        @server.on("send_chat")
        def broadcast_chat(sid, chat_text):
            eid = self.connected_sids.get(sid)
            player_struct = self.get_player_info(eid)
            chat_with_name = "<{}> {}".format(player_struct.name, chat_text)
            for sid, store in self.incoming_chats_store.items():
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
            # agent is 2 blocks high
            if (
                nx >= 0
                and ny >= 0
                and nz >= 0
                and nx < self.sl
                and ny < self.sl - 1
                and nz < self.sl
            ):
                if (
                    self.blocks[nx, ny, nz, 0] in PASSABLE_BLOCKS
                    and self.blocks[nx, ny + 1, nz, 0] in PASSABLE_BLOCKS
                ):
                    new_pos = Pos(x, y, z)
                    self.players[eid] = self.players[eid]._replace(pos=new_pos)

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
            nz_idms = [tuple(self.blocks[l].tolist()) for l in nz_locs]
            nz_locs = [(int(x), int(y), int(z)) for x, y, z in nz_locs]
            flattened_blocks = [nz_locs[i] + nz_idms[i] for i in range(len(nz_locs))]
            return flattened_blocks

        app = socketio.WSGIApp(server)
        eventlet.wsgi.server(eventlet.listen(("", port)), app)


if __name__ == "__main__":

    class Opt:
        pass

    spec = {"players": [], "mobs": [], "item_stacks": [], "coord_shift": (0, 0, 0), "agent": {}}
    world_opts = Opt()
    world_opts.sl = 32
    world_opts.world_server = True
    world_opts.port = 6001
    world = World(world_opts, spec)
