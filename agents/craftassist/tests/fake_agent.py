"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
from typing import List, Tuple

from droidlet.lowlevel.minecraft.mc_util import XYZ, IDM, Block
from droidlet.memory.memory_nodes import ChatNode
from .utils import Look, Pos, Item, Player
from agents.droidlet_agent import DroidletAgent
from droidlet.memory.craftassist.mc_memory import MCAgentMemory
from droidlet.memory.craftassist.mc_memory_nodes import VoxelObjectNode
from agents.craftassist.craftassist_agent import CraftAssistAgent
from droidlet.shared_data_structs import TICKS_PER_SEC, Time, MockOpt
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.map_to_dialogue_object import DialogueObjectMapper
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
from droidlet.perception.craftassist.rotation import look_vec, yaw_pitch
from droidlet.lowlevel.minecraft.mc_util import SPAWN_OBJECTS, get_locs_from_entity, fill_idmeta
from droidlet.lowlevel.minecraft import craftassist_specs
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import COLOR_BID_MAP
from droidlet.perception.craftassist import heuristic_perception

# how many internal, non-world-interacting steps agent takes before world steps:
WORLD_STEP = 10

WORLD_STEPS_PER_DAY = 480

HEAD_HEIGHT = 2


class FakeMCTime(Time):
    def __init__(self, world):
        self.world = world

    def get_world_hour(self):
        return (self.world.count % WORLD_STEPS_PER_DAY) / WORLD_STEPS_PER_DAY

    # converts from "seconds" to internal tick
    def round_time(self, t):
        return int(TICKS_PER_SEC * t)

    def get_time(self):
        return self.world.count * TICKS_PER_SEC

    def add_tick(self, ticks=1):
        for i in range(ticks):
            self.world.step()


class FakeCPPAction:
    NAME = "NULL"

    def __init__(self, agent):
        self.agent = agent

    def action(self, *args):
        self.agent.world_interaction_occurred = True

    def __call__(self, *args):
        if hasattr(self.agent, "recorder"):
            self.agent.recorder.record_action({"name": self.NAME, "args": list(args)})
        return self.action(*args)


class Dig(FakeCPPAction):
    NAME = "dig"

    def action(self, x, y, z):
        self.agent.world_interaction_occurred = True
        dug = self.agent.world.dig((x, y, z))
        if dug:
            self.agent._changed_blocks.append(((x, y, z), (0, 0)))
            return True
        else:
            return False


class SendChat(FakeCPPAction):
    NAME = "send_chat"

    def action(self, chat):
        self.agent.world_interaction_occurred = True
        logging.info("FakeAgent.send_chat: {}".format(chat))
        self.agent._outgoing_chats.append(chat)


class SetHeldItem(FakeCPPAction):
    NAME = "set_held_item"

    def action(self, arg):
        self.agent.world_interaction_occurred = True
        try:
            d, m = arg
            self.agent._held_item = (d, m)
        except TypeError:
            self.agent._held_item = (arg, 0)


class StepPosX(FakeCPPAction):
    NAME = "step_pos_x"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (1, 0, 0)


class StepNegX(FakeCPPAction):
    NAME = "step_neg_x"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (-1, 0, 0)


class StepPosZ(FakeCPPAction):
    NAME = "step_pos_z"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, 0, 1)


class StepNegZ(FakeCPPAction):
    NAME = "step_neg_z"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, 0, -1)


class StepPosY(FakeCPPAction):
    NAME = "step_pos_y"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, 1, 0)


class StepNegY(FakeCPPAction):
    NAME = "step_neg_y"

    def action(self):
        self.agent.world_interaction_occurred = True
        self.agent.pos += (0, -1, 0)


class StepForward(FakeCPPAction):
    NAME = "step_forward"

    def action(self):
        self.agent.world_interaction_occurred = True
        dx, dy, dz = self.agent._look_vec
        self.agent.pos += (dx, 0, dz)


class TurnAngle(FakeCPPAction):
    NAME = "turn_angle"

    def action(self, angle):
        self.agent.world_interaction_occurred = True
        if angle == 90:
            self.agent.turn_left()
        elif angle == -90:
            self.agent.turn_right()
        else:
            raise ValueError("bad angle={}".format(angle))


# FIXME!
class TurnLeft(FakeCPPAction):
    NAME = "turn_left"

    def action(self):
        self.agent.world_interaction_occurred = True
        old_l = (self.agent._look_vec[0], self.agent._look_vec[1])
        idx = self.agent.CCW_LOOK_VECS.index(old_l)
        new_l = self.agent.CCW_LOOK_VECS[(idx + 1) % len(self.agent.CCW_LOOK_VECS)]
        self.agent._look_vec[0] = new_l[0]
        self.agent._look_vec[2] = new_l[2]


# FIXME!
class TurnRight(FakeCPPAction):
    NAME = "turn_right"

    def action(self):
        self.agent.world_interaction_occurred = True
        old_l = (self.agent._look_vec[0], self.agent._look_vec[1])
        idx = self.agent.CCW_LOOK_VECS.index(old_l)
        new_l = self.agent.CCW_LOOK_VECS[(idx - 1) % len(self.agent.CCW_LOOK_VECS)]
        self.agent._look_vec[0] = new_l[0]
        self.agent._look_vec[2] = new_l[2]


class PlaceBlock(FakeCPPAction):
    NAME = "place_block"

    def action(self, x, y, z):
        self.agent.world_interaction_occurred = True
        block = ((x, y, z), self.agent._held_item)
        self.agent.world.place_block(block)
        self.agent._changed_blocks.append(block)
        return True


class LookAt(FakeCPPAction):
    NAME = "look_at"

    def action(self, x, y, z):
        self.agent.world_interaction_occurred = True
        look_vec = np.array(
            [x - self.agent.pos[0], y - self.agent.pos[1] - HEAD_HEIGHT, z - self.agent.pos[2]]
        )
        self.agent.set_look_vec(*look_vec.tolist())


class SetLook(FakeCPPAction):
    NAME = "set_look"

    def action(self, yaw, pitch):
        self.agent.world_interaction_occurred = True
        a = look_vec(yaw, pitch)
        self.agent.set_look_vec(a[0], a[1], a[2])


class Craft(FakeCPPAction):
    NAME = "craft"

    def action(self):
        raise NotImplementedError()


class FakeAgent(DroidletAgent):
    CCW_LOOK_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    default_frame = CraftAssistAgent.default_frame
    coordinate_transforms = CraftAssistAgent.coordinate_transforms

    def __init__(self, world, opts=None, do_heuristic_perception=False):
        self.head_height = HEAD_HEIGHT
        self.world = world
        self.chat_count = 0
        if not opts:
            opts = MockOpt()
        self.low_level_data = {
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
        super(FakeAgent, self).__init__(opts)
        self.do_heuristic_perception = do_heuristic_perception
        self.no_default_behavior = True
        self.last_task_memid = None
        pos = (0, 63, 0)
        if hasattr(self.world, "agent_data"):
            pos = self.world.agent_data["pos"]
        self.pos = np.array(pos, dtype="int")
        self.logical_form = None
        self.world_interaction_occurred = False
        self.world_interaction_occurred = False
        self._held_item: IDM = (0, 0)
        self._look_vec = (1, 0, 0)
        self._changed_blocks: List[Block] = []
        self._outgoing_chats: List[str] = []
        CraftAssistAgent.add_self_memory_node(self)

    def set_look_vec(self, x, y, z):
        l = np.array((x, y, z))
        l = l / np.linalg.norm(l)
        self._look_vec = (l[0], l[1], l[2])
        self.look = self.get_look()

    def init_perception(self):
        self.perception_modules = {}
        self.perception_modules["language_understanding"] = NSPQuerier(self.opts, self)
        self.perception_modules["low_level"] = LowLevelMCPerception(self, perceive_freq=1)
        self.perception_modules["heuristic"] = PerceptionWrapper(
            self, low_level_data=self.low_level_data
        )

    def init_physical_interfaces(self):
        self.dig = Dig(self)
        self.send_chat = SendChat(self)
        self.set_held_item = SetHeldItem(self)
        self.step_pos_x = StepPosX(self)
        self.step_neg_x = StepNegX(self)
        self.step_pos_z = StepPosZ(self)
        self.step_neg_z = StepNegZ(self)
        self.step_pos_y = StepPosY(self)
        self.step_neg_y = StepNegY(self)
        self.step_forward = StepForward(self)
        self.turn_angle = TurnAngle(self)
        self.turn_left = TurnLeft(self)
        self.turn_right = TurnRight(self)
        self.set_look = SetLook(self)
        self.look_at = LookAt(self)
        self.place_block = PlaceBlock(self)

    def init_memory(self):
        T = FakeMCTime(self.world)
        low_level_data = self.low_level_data.copy()
        low_level_data["check_inside"] = heuristic_perception.check_inside

        self.memory = MCAgentMemory(
            load_minecraft_specs=False,
            coordinate_transforms=self.coordinate_transforms,
            agent_time=T,
            agent_low_level_data=low_level_data,
        )
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
        }
        self.dialogue_manager = DialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            dialogue_object_mapper=DialogueObjectMapper,
            opts=self.opts,
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

    def controller_step(self):
        if self.logical_form is None:
            CraftAssistAgent.controller_step(self)
        else:  # logical form given directly:
            d = self.logical_form["logical_form"]
            chatstr = self.logical_form["chatstr"]
            speaker_name = self.logical_form["speaker"]
            chat_memid = self.memory.get_chat_id(
                self.memory.get_player_by_name(speaker_name).memid, chatstr
            )
            logical_form_memid = self.memory.add_logical_form(d)
            self.memory.add_triple(
                subj=chat_memid, pred_text="has_logical_form", obj=logical_form_memid
            )
            obj = self.dialogue_manager.dialogue_object_mapper.handle_logical_form(
                speaker_name, logical_form_memid
            )
            self.dialogue_manager.memory.untag(subj_memid=chat_memid, tag_text="unprocessed")
            if obj is not None:
                # TODO (interpreter): rethink this when interpreter is its own object
                if type(obj) is dict:
                    obj["task"](self, task_data=obj["data"])
                elif isinstance(obj, InterpreterBase):
                    obj.step(self)
                    if obj.finished:
                        self.memory.get_mem_by_id(obj.memid).finish()
                else:
                    raise Exception("strange obj returned from dialogue manager {}".format(obj))
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

    ########################
    ##  FAKE .PY METHODS  ##
    ########################

    def task_step(self):
        CraftAssistAgent.task_step(self, sleep_time=0)

    def point_at(*args):
        pass

    def perceive(self, force=False):
        # clear the chat buffer
        self.get_incoming_chats()
        # use the logical form as given...
        if self.logical_form:
            d = self.logical_form["logical_form"]
            chatstr = self.logical_form["chatstr"]
            speaker_name = self.logical_form["speaker"]
            chat_memid = self.memory.add_chat(
                self.memory.get_player_by_name(speaker_name).memid, chatstr
            )
            post_processed_parse = self.dialogue_manager.dialogue_object_mapper.postprocess_logical_form(
                speaker=speaker_name, chat=chatstr, logical_form=d
            )
            logical_form_memid = self.memory.add_logical_form(post_processed_parse)
            self.memory.add_triple(
                subj=chat_memid, pred_text="has_logical_form", obj=logical_form_memid
            )
            self.memory.tag(subj_memid=chat_memid, tag_text="unprocessed")
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

    ###################################
    ##  FAKE C++ PERCEPTION METHODS  ##
    ###################################

    def get_blocks(self, xa, xb, ya, yb, za, zb):
        return self.world.get_blocks(xa, xb, ya, yb, za, zb)

    def get_local_blocks(self, r):
        x, y, z = self.pos
        return self.get_blocks(x - r, x + r, y - r, y + r, z - r, z + r)

    def get_incoming_chats(self):
        c = self.chat_count
        self.chat_count = len(self.world.chat_log)
        return self.world.chat_log[c:].copy()

    def get_player(self):
        return Player(1, "fake_agent", Pos(*self.pos), self.get_look(), Item(*self._held_item))

    def get_mobs(self):
        return self.world.get_mobs()

    def get_item_stacks(self):
        return self.world.get_item_stacks()

    def get_other_players(self):
        return self.world.get_players()

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

        changes_to_be_updated = {"changed_block_attributes": {}}
        for xyz, idm in xyzbms:
            abs_xyz = tuple(np.array(xyz) + origin)
            self.perception_modules["low_level"].pending_agent_placed_blocks.add(abs_xyz)
            # TODO add force option so we don't need to make it as if agent placed
            interesting, player_placed, agent_placed = self.perception_modules[
                "low_level"
            ].mark_blocks_with_env_change(xyz, idm, boring_blocks)
            changes_to_be_updated["changed_block_attributes"][(abs_xyz, idm)] = [
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
        memid = self.memory.get_block_object_ids_by_xyz(abs_xyz)[0]
        for pred, obj in relations.items():
            self.memory.add_triple(subj=memid, pred_text=pred, obj_text=obj)
            # sooooorrry  FIXME? when we handle triples better in interpreter_helper
            if "has_" in pred:
                self.memory.tag(memid, obj)
        return self.memory.get_object_by_id(memid)

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
        C = self.world.to_world_coords(c)
        A = self.world.to_world_coords(self.pos)
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
            p = np.round(np.array(self.world.to_world_coords(mob.pos)))
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
    ):
        class NubWorld:
            def __init__(self):
                self.count = 0

        super().__init__(NubWorld(), opts=opts, do_heuristic_perception=do_heuristic_perception)
        # if active is set to false, the fake player's step is passed.
        self.active = active
        self.get_world_pos = get_world_pos
        if struct:
            self.entityId = struct.entityId
            self.name = struct.name
            self.look = struct.look
            self.mainHand = struct.mainHand
            self.pos = np.array((struct.pos.x, struct.pos.y, struct.pos.z))
        else:
            self.entityId = np.random.randint(0, 10000000)
            # FIXME
            self.name = str(self.entityId)
            self.look = Look(270, 0)
            self.mainHand = Item(0, 0)
        if name:
            self.name = name
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

    def controller_step(self):
        if self.logical_form is None:
            CraftAssistAgent.controller_step(self)
            query = "SELECT MEMORY FROM Task WHERE ((prio >= 0) AND (paused <= 0))"
            _, task_mems = self.memory.basic_search(query)
            if not task_mems:
                if len(self.lf_list) > 0:
                    self.logical_form = self.lf_list[0]
                    del self.lf_list[0]
        else:  # logical form given directly:
            d = self.logical_form["logical_form"]
            chatstr = self.logical_form["chatstr"]
            speaker_name = self.logical_form["speaker"]
            chat_memid = self.memory.get_chat_id(
                self.memory.get_player_by_name(speaker_name).memid, chatstr
            )
            logical_form_memid = self.memory.add_logical_form(d)
            self.memory.add_triple(
                subj=chat_memid, pred_text="has_logical_form", obj=logical_form_memid
            )
            obj = self.dialogue_manager.dialogue_object_mapper.handle_logical_form(
                speaker_name, logical_form_memid
            )
            self.dialogue_manager.memory.untag(subj_memid=chat_memid, tag_text="unprocessed")
            if obj is not None:
                # TODO (interpreter): rethink this when interpreter is its own object
                if type(obj) is dict:
                    obj["task"](self, task_data=obj["data"])
                elif isinstance(obj, InterpreterBase):
                    obj.step(self)
                    if obj.finished:
                        self.memory.get_mem_by_id(obj.memid).finish()
                else:
                    raise Exception("strange obj returned from dialogue manager {}".format(obj))
            self.logical_form = None

    def get_info(self):
        return Player(
            self.entityId,
            self.name,
            Pos(self.pos[0], self.pos[1], self.pos[2]),
            self.look,
            self.mainHand,
        )

    # fake player does not respond to chats, etc.
    # fixme for clarifications?
    def get_incoming_chats(self):
        return []

    def set_lf_list(self, lf_list):
        self.lf_list = lf_list

    def add_to_world(self, world):
        self.world = world
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
        self.world.players.append(self)
