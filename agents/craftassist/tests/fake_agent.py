"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging

import numpy as np
from typing import List, Tuple

from droidlet.lowlevel.minecraft.mc_util import XYZ, IDM, Block
from droidlet.memory.memory_nodes import ChatNode
from droidlet.base_util import Look, Pos
from droidlet.shared_data_struct.craftassist_shared_utils import Item, Slot, Player, ItemStack
from agents.droidlet_agent import DroidletAgent
from droidlet.memory.craftassist.mc_memory import MCAgentMemory
from droidlet.memory.craftassist.mc_memory_nodes import TripleNode, VoxelObjectNode
from agents.craftassist.craftassist_agent import CraftAssistAgent
from droidlet.shared_data_structs import MockOpt
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.lowlevel.minecraft.shapes import SPECIAL_SHAPE_FNS
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    BORING_BLOCKS,
    PASSABLE_BLOCKS,
)
from droidlet.dialog.craftassist.mc_dialogue_task import MCBotCapabilities
from droidlet.interpreter import InterpreterBase
from droidlet.interpreter.craftassist import (
    MCGetMemoryHandler,
    PutMemoryHandler,
    MCInterpreter,
    dance,
)
from droidlet.perception.craftassist.low_level_perception import LowLevelMCPerception
from droidlet.perception.craftassist.heuristic_perception import PerceptionWrapper
from droidlet.shared_data_struct.rotation import yaw_pitch
from droidlet.lowlevel.minecraft.mc_util import SPAWN_OBJECTS, get_locs_from_entity, fill_idmeta
from droidlet.lowlevel.minecraft import craftassist_specs
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import COLOR_BID_MAP
from droidlet.perception.craftassist import heuristic_perception
from droidlet.shared_data_struct.craftassist_shared_utils import CraftAssistPerceptionData
from droidlet.lowlevel.minecraft.pyworld.physical_interfaces import (
    FakeMCTime,
    init_agent_interfaces,
    WORLD_STEP,
    WORLD_STEPS_PER_DAY,
    HEAD_HEIGHT,
)


def default_low_level_data():
    low_level_data = {
        "mobs": SPAWN_OBJECTS,
        "mob_property_data": craftassist_specs.get_mob_property_data(),
        "schematics": craftassist_specs.get_schematics(),
        "block_data": craftassist_specs.get_block_data(),
        "block_property_data": craftassist_specs.get_block_property_data(),
        "color_data": craftassist_specs.get_colour_data(),
        "boring_blocks": BORING_BLOCKS,
        "passable_blocks": PASSABLE_BLOCKS,
        "fill_idmeta": fill_idmeta,
        "color_bid_map": COLOR_BID_MAP,
    }
    return low_level_data


class FakeAgent(DroidletAgent):
    default_frame = CraftAssistAgent.default_frame
    coordinate_transforms = CraftAssistAgent.coordinate_transforms

    def __init__(
        self,
        world,
        opts=None,
        do_heuristic_perception=False,
        prebuilt_perception=None,
        low_level_data=None,
        use_place_field=True,
        prebuilt_db=None,
    ):
        self.backend = "pyworld"
        self.prebuilt_db = prebuilt_db
        self.use_place_field = use_place_field
        self.mark_airtouching_blocks = do_heuristic_perception
        self.head_height = HEAD_HEIGHT
        self.mainHand = getattr(self, "mainHand", Item(0, 0))
        self.look = getattr(self, "look", Look(270, 0))
        self.add_to_world(world)
        # do not allow the world to step this agent; this agent steps the world.
        self.no_step_from_world = True
        self.chat_count = 0
        # use these to not have to re-init models if running many tests:
        self.prebuilt_perception = prebuilt_perception
        if not opts:
            opts = MockOpt()
        self.e2e_mode = getattr(opts, "e2e_mode", False)
        if low_level_data is None:
            low_level_data = default_low_level_data()
        self.low_level_data = low_level_data
        super(FakeAgent, self).__init__(opts, name=getattr(self, "name", None))
        self.do_heuristic_perception = do_heuristic_perception
        self.no_default_behavior = True
        self.last_task_memid = None
        self.logical_form = None
        self.world_interaction_occurred = False
        self._held_item: IDM = (0, 0)
        self._look_vec = (1, 0, 0)
        self._changed_blocks: List[Block] = []
        self._outgoing_chats: List[str] = []
        CraftAssistAgent.add_self_memory_node(self)

    def add_to_world(self, world):
        self.world = world
        self.entityId = getattr(self, "entityId", world.new_eid())

        pos = (0, 63, 0)
        if hasattr(self.world, "agent_data"):
            pos = self.world.agent_data["pos"]
        self.pos = getattr(self, "pos", np.array(pos, dtype="int"))
        self.world.players[self.entityId] = self

    def set_look_vec(self, x, y, z):
        l = np.array((x, y, z))
        l = l / np.linalg.norm(l)
        self._look_vec = (l[0], l[1], l[2])
        self.look = self.get_look()

    def init_perception(self):
        self.perception_modules = {}
        if self.prebuilt_perception:
            for k, v in self.prebuilt_perception:
                self.perception_modules[k] = v
                # make a method to do this....
                self.perception_modules[k].agent = self
        else:
            self.perception_modules["language_understanding"] = NSPQuerier(self.opts, self)
        self.perception_modules["low_level"] = LowLevelMCPerception(self, perceive_freq=1)
        self.perception_modules["heuristic"] = PerceptionWrapper(
            self,
            low_level_data=self.low_level_data,
            mark_airtouching_blocks=self.mark_airtouching_blocks,
        )

    def init_physical_interfaces(self):
        init_agent_interfaces(self)

    def init_memory(self):
        T = FakeMCTime(self.world)
        low_level_data = self.low_level_data.copy()
        low_level_data["check_inside"] = heuristic_perception.check_inside
        kwargs = {
            "load_minecraft_specs": False,
            "coordinate_transforms": self.coordinate_transforms,
            "agent_time": T,
            "agent_low_level_data": low_level_data,
        }
        if not self.use_place_field:
            kwargs["place_field_pixels_per_unit"] = -1
        if self.prebuilt_db is not None:
            kwargs["copy_from_backup"] = self.prebuilt_db
        self.memory = MCAgentMemory(**kwargs)
        # Add dances to memory
        dance.add_default_dances(self.memory)

    def init_controller(self):
        dialogue_object_classes = {}
        dialogue_object_classes["bot_capabilities"] = {"task": MCBotCapabilities, "data": {}}
        dialogue_object_classes["interpreter"] = MCInterpreter
        dialogue_object_classes["get_memory"] = MCGetMemoryHandler
        dialogue_object_classes["put_memory"] = PutMemoryHandler
        low_level_interpreter_data = {
            "block_data": craftassist_specs.get_block_data(),
            "special_shape_functions": SPECIAL_SHAPE_FNS,
            "color_bid_map": self.low_level_data["color_bid_map"],
            "get_all_holes_fn": heuristic_perception.get_all_nearby_holes,
            "get_locs_from_entity": get_locs_from_entity,
            "allow_clarification": False,
        }
        self.dialogue_manager = DialogueManager(
            self.memory,
            dialogue_object_classes,
            self.opts,
            low_level_interpreter_data=low_level_interpreter_data,
        )

    def set_logical_form(self, lf, chatstr, speaker):
        self.logical_form = {"logical_form": lf, "chatstr": chatstr, "speaker": speaker}

    def step(self):
        if hasattr(self.world, "step"):
            if self.world_interaction_occurred or self.count % WORLD_STEP == 0:
                self.world.step()
                self.world_interaction_occurred = False
        if hasattr(self, "recorder"):
            self.recorder.record_world()
        super().step()

    def perceive(self, force=False):
        if self.e2e_mode:
            perception_output = self.perception_modules["low_level"].perceive(force=force)
            self.areas_to_perceive = self.memory.update(perception_output, self.areas_to_perceive)[
                "areas_to_perceive"
            ]
            super().perceive(force=force)
            if "semseg" in self.perception_modules:
                sem_seg_perception_output = self.perception_modules["semseg"].perceive()
                self.memory.update(sem_seg_perception_output)
            return
        # clear the chat buffer
        self.get_incoming_chats()
        if self.logical_form:  # use the logical form as given...
            DroidletAgent.process_language_perception(
                self,
                self.logical_form["speaker"],
                self.logical_form["chatstr"],
                self.logical_form["chatstr"],
                self.logical_form["logical_form"],
            )
            force = True
        perception_output = self.perception_modules["low_level"].perceive(force=force)
        self.areas_to_perceive = self.memory.update(perception_output, self.areas_to_perceive)[
            "areas_to_perceive"
        ]
        if self.do_heuristic_perception:
            if force or not self.agent.memory.task_stack_peek():
                # perceive from heuristic perception module
                heuristic_perception_output = self.perception_modules["heuristic"].perceive()
                self.memory.update(heuristic_perception_output)

    def controller_step(self):
        CraftAssistAgent.controller_step(self)
        # if the logical form was set explicitly, clear it, so that it won't keep
        # being perceived and used to respawn new interpreters
        self.logical_form = None

    def setup_test(self):
        self.task_steps_count = 0

    def clear_outgoing_chats(self):
        self._outgoing_chats.clear()

    def get_last_outgoing_chat(self):
        try:
            return self._outgoing_chats[-1]
        except IndexError:
            return None

    def task_step(self):
        CraftAssistAgent.task_step(self, sleep_time=0)

    def point_at(*args):
        pass

    def get_blocks(self, xa, xb, ya, yb, za, zb):
        return self.world.get_blocks(xa, xb, ya, yb, za, zb)

    def get_info(self):
        return Player(
            self.entityId,
            self.name,
            Pos(self.pos[0], self.pos[1], self.pos[2]),
            self.look,
            self.mainHand,
        )

    def get_local_blocks(self, r):
        x, y, z = self.pos
        return self.get_blocks(x - r, x + r, y - r, y + r, z - r, z + r)

    def get_incoming_chats(self):
        c = self.chat_count
        self.chat_count = len(self.world.chat_log)
        return self.world.chat_log[c:].copy()

    def get_player(self):
        return Player(
            self.entityId, "fake_agent", Pos(*self.pos), self.get_look(), Item(*self._held_item)
        )

    def get_mobs(self):
        return self.world.get_mobs()

    # FIXME! use pyworld_mover !!!!
    def get_item_stacks(self, holder_entityId=-1, get_all=False):
        """
        by default
        only return items not in any agent's inventory, matching cuberite
        returns a list of [ItemStack, holder_id] pairs
        """
        items = self.world.get_items()
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

        return self.world.get_item_stacks()

    def get_other_players(self):
        all_players = self.world.get_players()
        return [p for p in all_players if p.entityId != self.entityId]

    def get_other_player_by_name(self):
        raise NotImplementedError()

    def get_vision(self):
        raise NotImplementedError()

    def get_line_of_sight(self):
        raise NotImplementedError()

    def get_look(self):
        yaw, pitch = yaw_pitch(self._look_vec)
        return Look(yaw, pitch)

    def get_player_line_of_sight(self, player_struct):
        if hasattr(self.world, "get_line_of_sight"):
            pos = (player_struct.pos.x, player_struct.pos.y + HEAD_HEIGHT, player_struct.pos.z)
            pitch = player_struct.look.pitch
            yaw = player_struct.look.yaw
            xsect = self.world.get_line_of_sight(pos, yaw, pitch)
            if xsect is not None:
                return Pos(*xsect)
        else:
            raise NotImplementedError()

    def get_changed_blocks(self) -> List[Block]:
        # need a better solution here
        r = self._changed_blocks.copy()
        self._changed_blocks.clear()
        return r

    def safe_get_changed_blocks(self) -> List[Block]:
        return self.get_changed_blocks()

    ######################################
    ## World setup
    ######################################

    def set_blocks(self, xyzbms: List[Block], boring_blocks: Tuple[int], origin: XYZ = (0, 0, 0)):
        """Change the state of the world, block by block,
        store in memory"""

        changes_to_be_updated = CraftAssistPerceptionData(changed_block_attributes={})
        for xyz, idm in xyzbms:
            abs_xyz = tuple(np.array(xyz) + origin)
            self.perception_modules["low_level"].pending_agent_placed_blocks.add(abs_xyz)
            # TODO add force option so we don't need to make it as if agent placed
            interesting, player_placed, agent_placed = self.perception_modules[
                "low_level"
            ].mark_blocks_with_env_change(xyz, idm, boring_blocks)
            changes_to_be_updated.changed_block_attributes[(abs_xyz, idm)] = [
                interesting,
                player_placed,
                agent_placed,
            ]
            self.world.place_block((abs_xyz, idm))
        # TODO: to be named to normal update function
        self.memory.update(changes_to_be_updated, self.areas_to_perceive)

    def add_object(
        self, xyzbms: List[Block], origin: XYZ = (0, 0, 0), relations={}
    ) -> VoxelObjectNode:
        """Add an object to memory as if it was placed block by block

        Args:
        - xyzbms: a list of relative (xyz, idm)
        - origin: (x, y, z) of the corner

        Returns an VoxelObjectNode
        """
        boring_blocks = self.low_level_data["boring_blocks"]
        self.set_blocks(xyzbms, boring_blocks, origin)
        abs_xyz = tuple(np.array(xyzbms[0][0]) + origin)
        memid = self.memory.get_object_info_by_xyz(abs_xyz, "BlockObjects")[0]
        for pred, obj in relations.items():
            self.memory.nodes[TripleNode.NODE_TYPE].create(
                self.memory, subj=memid, pred_text=pred, obj_text=obj
            )
            # sooooorrry  FIXME? when we handle triples better in interpreter_helper
            if "has_" in pred:
                self.memory.nodes[TripleNode.NODE_TYPE].tag(self.memory, memid, obj)
        return self.memory.get_mem_by_id(memid)

    # WARNING!! this does not step the world, but directly fast-forwards
    # to count.  Use only in world setup, once world is running!
    def add_object_ff_time(
        self, count, xyzbms: List[Block], origin: XYZ = (0, 0, 0), relations={}
    ) -> VoxelObjectNode:
        assert count >= self.world.count
        self.world.set_count(count)
        return self.add_object(xyzbms, origin, relations=relations)

    ######################################
    ## visualization
    ######################################

    def draw_slice(self, h=None, r=5, c=None):
        if not h:
            h = self.pos[1]
        if c:
            c = [c[0], h, c[1]]
        else:
            c = [self.pos[0], h, self.pos[2]]
        C = self.world.to_npy_coords(c)
        A = self.world.to_npy_coords(self.pos)
        shifted_agent_pos = [A[0] - C[0] + r, A[2] - C[2] + r]
        npy = self.world.get_blocks(
            c[0] - r, c[0] + r, c[1], c[1], c[2] - r, c[2] + r, transpose=False
        )
        npy = npy[:, 0, :, 0]
        try:
            npy[shifted_agent_pos[0], shifted_agent_pos[1]] = 1024
        except:
            pass
        mobnums = {"rabbit": -1, "cow": -2, "pig": -3, "chicken": -4, "sheep": -5}
        nummobs = {-1: "rabbit", -2: "cow", -3: "pig", -4: "chicken", -5: "sheep"}
        for mob in self.world.mobs:
            # todo only in the plane?
            p = np.round(np.array(self.world.to_npy_coords(mob.pos)))
            p = p - C
            try:
                npy[p[0] + r, p[1] + r] = mobnums[mob.mobname]
            except:
                pass
        mapslice = ""
        height = npy.shape[0]
        width = npy.shape[1]

        def xs(x):
            return x + int(self.pos[0]) - r

        def zs(z):
            return z + int(self.pos[2]) - r

        mapslice = mapslice + " " * (width + 2) * 3 + "\n"
        for i in reversed(range(height)):
            mapslice = mapslice + str(xs(i)).center(3)
            for j in range(width):
                if npy[i, j] > 0:
                    if npy[i, j] == 1024:
                        mapslice = mapslice + " A "
                    else:
                        mapslice = mapslice + str(npy[i, j]).center(3)
                elif npy[i, j] == 0:
                    mapslice = mapslice + " * "
                else:
                    npy[i, j] = mapslice + " " + nummobs[npy[i, j]][0] + " "
            mapslice = mapslice + "\n"
            mapslice = mapslice + "   "
            for j in range(width):
                mapslice = mapslice + " * "
            mapslice = mapslice + "\n"
        mapslice = mapslice + "   "
        for j in range(width):
            mapslice = mapslice + str(zs(j)).center(3)

        return mapslice


class FakePlayer(FakeAgent):
    """
    a fake player that can do actions, but does not currently interact with agent.
    """

    def __init__(
        self,
        struct=None,
        opts=(),
        do_heuristic_perception=False,
        get_world_pos=False,
        name="",
        active=True,
        low_level_data=None,
        use_place_field=False,
        prebuilt_db=None,
    ):
        class NubWorld:
            def __init__(self):
                self.is_nubworld = True
                self.count = 0

        if struct:
            self.entityId = struct.entityId
            self.name = struct.name
            self.look = struct.look
            self.mainHand = struct.mainHand
            self.pos = np.array((struct.pos.x, struct.pos.y, struct.pos.z))
        else:
            self.entityId = int(np.random.randint(0, 10000000))
            # FIXME
            self.name = str(self.entityId)
            self.look = Look(270, 0)
            self.mainHand = Item(0, 0)
            self.pos = None
        if name:
            self.name = name
        self.get_world_pos = get_world_pos
        super().__init__(
            NubWorld(),
            opts=opts,
            do_heuristic_perception=do_heuristic_perception,
            low_level_data=low_level_data,
            use_place_field=use_place_field,
            prebuilt_db=prebuilt_db,
        )
        # if active is set to false, the fake player's step is passed.
        self.active = active
        # the world steps this agent
        self.no_step_from_world = False
        self.lf_list = []
        self.look_towards_move = True

        def get_recent_chat():
            m = self.memory
            if self.lf_list:
                C = ChatNode(m, ChatNode.create(m, self.name, self.lf_list[0]["chatstr"]))
            else:
                C = ChatNode(m, ChatNode.create(self.memory, self.name, ""))
            return C

        self.memory.get_most_recent_incoming_chat = get_recent_chat

    def step(self):
        if self.active:
            DroidletAgent.step(self)

    # fake player does not respond to chats, etc.
    # fixme for clarifications?
    def get_incoming_chats(self):
        return []

    def set_lf_list(self, lf_list):
        self.lf_list = lf_list

    def add_to_world(self, world):
        self.world = world
        if getattr(world, "is_nubworld", False):
            return
        if self.pos is None or self.get_world_pos:
            xz = np.random.randint(0, world.sl, (2,))
            slice = self.world.blocks[xz[0], :, xz[1], 0]
            nz = np.flatnonzero(slice)
            if len(nz) == 0:
                # player will be floating, but why no floor here?
                h = 0
            else:
                # if top block is nonzero player will be trapped
                h = nz[-1]
            off = self.world.coord_shift
            self.pos = np.array(
                (float(xz[0]) + off[0], float(h + 1) + off[1], float(xz[1]) + off[2]), dtype="int"
            )
        self.world.players[self.entityId] = self
