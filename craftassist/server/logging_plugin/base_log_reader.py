"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import binascii

from decoder import Decoder
import hooks
import util


# https://api.cuberite.org/Globals.html
dtAttack = 0
etMob = 4


class BaseLogReader:
    def __init__(self, logdir):
        self.logdir = logdir
        fp = open(logdir + "/logging.bin", "rb")
        self.decoder = Decoder(fp)
        self.last_tick = -1
        self.player_eids = set()

    def start(self):
        version_major = self.decoder.readShort()
        version_minor = self.decoder.readShort()
        print("Version: {}.{}".format(version_major, version_minor))

        while True:
            try:
                buf_start = self.decoder.count
                hook_id = self.decoder.readByte()
                world_tick = self.decoder.readLong()
                if world_tick < self.last_tick:
                    raise RuntimeError(
                        "Error: {} < {}\n".format(world_tick, self.last_tick)
                        + "buf_start={} hook_id={}".format(buf_start, hook_id)
                    )
                self.last_tick = world_tick
                self.decode_and_handle_hook(hook_id, world_tick, buf_start)
            except EOFError:
                return

    def decode_and_handle_hook(self, hook_id, world_tick, buf_start):
        args = [world_tick, buf_start]

        if hook_id == hooks.WORLD_STARTED:
            # Check settings.ini hash
            expected_settings_hash = binascii.hexlify(self.decoder.readRaw(20)).decode("ascii")
            settings_hashes = util.get_hashes(self.logdir + "/settings.ini")
            assert (
                expected_settings_hash in settings_hashes
            ), "Bad hash for settings.ini: {} not in {}".format(
                expected_settings_hash, settings_hashes
            )
            # Check world.ini hash
            expected_world_hash = binascii.hexlify(self.decoder.readRaw(20)).decode("ascii")
            world_hashes = util.get_hashes(self.logdir + "/world/world.ini")
            assert (
                expected_world_hash in world_hashes
            ), "Bad hash for world/world.ini: {} not in {}".format(
                expected_world_hash, world_hashes
            )

        elif hook_id == hooks.PLAYER_SPAWNED:
            eid = self.decoder.readLong()
            name = self.decoder.readString()
            pos = self.decoder.readFloatPos()
            look = self.decoder.readLook()
            args += [eid, name, pos, look]

            # FIXME: remove when v0.2 patch no longer needed
            self.player_eids.add(eid)

        elif hook_id == hooks.PLAYER_DESTROYED:
            eid = self.decoder.readLong()
            args += [eid]

        elif hook_id == hooks.PLAYER_MOVING:
            eid = self.decoder.readLong()
            oldpos = self.decoder.readFloatPos()
            newpos = self.decoder.readFloatPos()
            args += [eid, oldpos, newpos]

        elif hook_id == hooks.CHUNK_AVAILABLE:
            cx, cz = self.decoder.readLong(), self.decoder.readLong()
            args += [cx, cz]

        elif hook_id == hooks.BLOCK_SPREAD:
            pos = self.decoder.readIntPos()
            source = self.decoder.readByte()
            args += [pos, source]

        elif hook_id == hooks.CHAT:
            eid = self.decoder.readLong()
            chat = self.decoder.readString()
            args += [eid, chat]

        elif hook_id == hooks.COLLECTING_PICKUP:
            eid = self.decoder.readLong()
            item = self.decoder.readItem()
            args += [eid, item]

        elif hook_id == hooks.KILLED:
            eid = self.decoder.readLong()
            args += [eid]

        elif hook_id == hooks.PLAYER_BROKEN_BLOCK:
            eid = self.decoder.readLong()
            pos = self.decoder.readIntPos()
            face = self.decoder.readByte()
            block = self.decoder.readBlock()
            args += [eid, pos, face, block]

        elif hook_id == hooks.PLAYER_PLACED_BLOCK:
            eid = self.decoder.readLong()
            pos = self.decoder.readIntPos()
            block = self.decoder.readBlock()
            args += [eid, pos, block]

        elif hook_id == hooks.PLAYER_USED_BLOCK:
            eid = self.decoder.readLong()
            pos = self.decoder.readIntPos()
            face = self.decoder.readByte()
            cursor = [self.decoder.readFloat() for _ in range(3)]
            block = self.decoder.readBlock()
            args += [eid, pos, face, cursor, block]

        elif hook_id == hooks.PLAYER_USED_ITEM:
            eid = self.decoder.readLong()
            pos = self.decoder.readIntPos()
            face = self.decoder.readByte()
            cursor = [self.decoder.readFloat() for _ in range(3)]
            item = self.decoder.readShort()
            args += [eid, pos, face, cursor, item]

        elif hook_id == hooks.POST_CRAFTING:
            eid = self.decoder.readLong()
            grid_h, grid_w = self.decoder.readByte(), self.decoder.readByte()
            grid = [self.decoder.readItem() for _ in range(grid_h * grid_w)]
            recipe_h, recipe_w = self.decoder.readByte(), self.decoder.readByte()
            recipe = [self.decoder.readItem() for _ in range(recipe_h * recipe_w)]
            result = self.decoder.readItem()
            args += [eid, (grid_h, grid_w, grid), (recipe_h, recipe_w, recipe), result]

        elif hook_id == hooks.SPAWNED_ENTITY:
            eid = self.decoder.readLong()
            etype = self.decoder.readByte()
            pos = self.decoder.readFloatPos()
            look = self.decoder.readLook()
            args += [eid, etype, pos, look]
            if etype == etMob:
                mtype = self.decoder.readByte()
                args += [mtype]

        elif hook_id == hooks.SPAWNED_MONSTER:
            eid = self.decoder.readLong()
            etype = self.decoder.readByte()
            mobtype = self.decoder.readByte()
            pos = self.decoder.readFloatPos()
            look = self.decoder.readLook()
            args += [eid, etype, mobtype, pos, look]

        elif hook_id == hooks.TAKE_DAMAGE:
            eid = self.decoder.readLong()
            dmgType = self.decoder.readByte()
            finalDmg = self.decoder.readDouble()
            rawDmg = self.decoder.readDouble()
            knockback = self.decoder.readFloatPos()
            args += [eid, dmgType, finalDmg, rawDmg, knockback]
            if dmgType == dtAttack:
                attackerId = self.decoder.readLong()
                args += [attackerId]

        elif hook_id == hooks.WEATHER_CHANGED:
            weather = self.decoder.readByte()
            args += [weather]

        elif hook_id == hooks.MONSTER_MOVED:
            eid = self.decoder.readLong()

            # patch for broken v0.2, where MONSTER_MOVED and PLAYER_LOOK have
            # the same hook_id
            if eid in self.player_eids:
                hook_id = hooks.PLAYER_LOOK
                look = self.decoder.readLook()
                args += [eid, look]
            else:
                pos = self.decoder.readFloatPos()
                look = self.decoder.readLook()
                args += [eid, pos, look]

        elif hook_id == hooks.PLAYER_LOOK:
            eid = self.decoder.readLong()
            look = self.decoder.readLook()
            args += [eid, look]

        else:
            print("Debug:", args)
            raise NotImplementedError("Not implemented: hook id {}".format(hook_id))

        # Call subclass handler method
        # e.g. for PLAYER_SPAWNED, call self.on_player_spawned
        func_name = "on_" + util.get_hook_name(hook_id).lower()
        func = getattr(self, func_name, lambda *args: None)
        func(*args)
