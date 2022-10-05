"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import logging
import faulthandler
import signal
import random
import sentry_sdk
import time
import json
import numpy as np
from multiprocessing import set_start_method
from collections import namedtuple
from datetime import datetime, timedelta
from copy import deepcopy

# `from craftassist.agent` instead of `from .` because this file is
# also used as a standalone script and invoked via `python craftassist_agent.py`
from droidlet.interpreter.craftassist import default_behaviors, dance
from droidlet.memory.craftassist import mc_memory
from droidlet.memory.memory_nodes import ChatNode, SelfNode
from droidlet.shared_data_struct import rotation
from droidlet.lowlevel.minecraft.craftassist_mover import (
    CraftassistMover,
    from_minecraft_look_to_droidlet,
    from_minecraft_xyz_to_droidlet,
)
from droidlet.lowlevel.minecraft.pyworld_mover import PyWorldMover

from droidlet.lowlevel.minecraft.shapes import SPECIAL_SHAPE_FNS
import droidlet.dashboard as dashboard

if __name__ == "__main__":
    # this line has to go before any imports that contain @sio.on functions
    # or else, those @sio.on calls become no-ops
    print("starting dashboard...")
    dashboard.start()

from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.dialog.dialogue_task import build_question_json
from droidlet.base_util import Pos, Look, npy_to_blocks_list
from droidlet.shared_data_struct.craftassist_shared_utils import Player, Item
from agents.droidlet_agent import DroidletAgent
from droidlet.perception.semantic_parsing.nsp_querier import NSPQuerier
from agents.argument_parser import ArgumentParser
from droidlet.dialog.craftassist.mc_dialogue_task import MCBotCapabilities
from droidlet.interpreter.craftassist import MCGetMemoryHandler, PutMemoryHandler, MCInterpreter
from droidlet.perception.craftassist.low_level_perception import LowLevelMCPerception
from droidlet.perception.craftassist.manual_edits_perception import ManualChangesPerception
from droidlet.lowlevel.minecraft.mc_util import (
    cluster_areas,
    MCTime,
    SPAWN_OBJECTS,
    get_locs_from_entity,
    fill_idmeta,
)
from droidlet.perception.craftassist.detection_model_perception import DetectionWrapper
from droidlet.lowlevel.minecraft import craftassist_specs
from droidlet.lowlevel.minecraft.craftassist_cuberite_utils.block_data import (
    COLOR_BID_MAP,
    BORING_BLOCKS,
    PASSABLE_BLOCKS,
)
from droidlet.lowlevel.minecraft import shape_util
from droidlet.perception.craftassist import heuristic_perception
from droidlet.tools.artifact_scripts.try_download import try_download_artifacts
from droidlet.event import sio

faulthandler.register(signal.SIGUSR1)

random.seed(0)
log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().handlers.clear()

sentry_sdk.init()  # enabled if SENTRY_DSN set in env
DEFAULT_BEHAVIOUR_TIMEOUT = 20
DEFAULT_FRAME = "SPEAKER"


class CraftAssistAgent(DroidletAgent):
    default_frame = DEFAULT_FRAME
    coordinate_transforms = rotation

    def __init__(self, opts):
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
        self.allow_clarification = opts.allow_clarification
        self.backend = opts.backend
        self.mark_airtouching_blocks = opts.mark_airtouching_blocks
        super(CraftAssistAgent, self).__init__(opts)
        self.no_default_behavior = opts.no_default_behavior
        self.agent_type = "craftassist"
        self.point_targets = []
        self.last_chat_time = 0
        self.dash_enable_map = False # dash has map disabled by default
        self.map_last_updated = datetime.now()
        # areas must be perceived at each step
        # List of tuple (XYZ, radius), each defines a cube
        self.areas_to_perceive = []
        self.perceive_on_chat = True
        self.add_self_memory_node()
        self.init_event_handlers()

        shape_util_dict = {
            "shape_names": shape_util.SHAPE_NAMES,
            "shape_option_fn_map": shape_util.SHAPE_OPTION_FUNCTION_MAP,
            "bid": shape_util.bid(),
            "shape_fns": shape_util.SHAPE_FNS,
        }
        # list of (prob, default function) pairs
        self.visible_defaults = [
            (0.001, (default_behaviors.build_random_shape, shape_util_dict)),
            (0.005, default_behaviors.come_to_player),
        ]

    def get_chats(self):
        """This function is a wrapper around self.mover.get_incoming_chats and adds a new
        chat self.dashboard_chat which is set by the dashboard."""
        all_chats = self.mover.get_incoming_chats()
        updated_chats = []
        if self.dashboard_chat:
            updated_chats.append(self.dashboard_chat)
            self.dashboard_chat = None
        updated_chats.extend(all_chats)
        return updated_chats

    def get_all_players(self):
        """This function is a wrapper around self.mover.get_other_players and adds a new
        player called "dashboard" if it doesn't already exist."""
        all_players = self.mover.get_other_players()
        updated_players = all_players
        player_exists = False
        for player in all_players:
            if player.name == "dashboard":
                player_exists = True
        if not player_exists:
            if self.backend == "cuberite":
                newPlayer = Player(
                    12345678, "dashboard", Pos(0.0, 64.0, 0.0), Look(0.0, 0.0), Item(0, 0)
                )
            elif self.backend == "pyworld":
                # FIXME this won't be updated with actual player position until/unless the player moves (abs_move)
                newPlayer = Player(
                    12345678, "dashboard", Pos(0.0, 5.0, 0.0), Look(0.0, 0.0), Item(0, 0)
                )
            updated_players.append(newPlayer)
        return updated_players

    def get_all_player_line_of_sight(self, player_struct):
        """return a fixed value for "dashboard" player"""
        # FIXME, this is too dangerous.
        if player_struct.name == "dashboard":
            if self.backend == "cuberite":
                return Pos(-1, 63, 14)
        return self.mover.get_player_line_of_sight(player_struct)

    def init_event_handlers(self):
        """Handle the socket events"""
        super().init_event_handlers()

        @sio.on("getVoxelWorldInitialState")
        def setup_agent_initial_state(sid):
            MAX_RADIUS = 50
            logging.info("in setup_world_initial_state")
            agent_pos = self.get_player().pos
            x, y, z = round(agent_pos.x), round(agent_pos.y), round(agent_pos.z)
            origin = (x - MAX_RADIUS, y - MAX_RADIUS, z - MAX_RADIUS)
            yzxb = self.get_blocks(
                x - MAX_RADIUS,
                x + MAX_RADIUS,
                y - MAX_RADIUS,
                y + MAX_RADIUS,
                z - MAX_RADIUS,
                z + MAX_RADIUS,
            )
            blocks = npy_to_blocks_list(yzxb, origin=origin)
            blocks = [
                ((int(xyz[0]), int(xyz[1]), int(xyz[2])), (int(idm[0]), int(idm[1])))
                for xyz, idm in blocks
            ]
            payload = {
                "status": "setupWorldInitialState",
                "world_state": {
                    "agent": {
                        "name": "agent",
                        "x": float(agent_pos.x),
                        "y": float(agent_pos.y),
                        "z": float(agent_pos.z),
                    },
                    "block": blocks,
                },
            }
            sio.emit("setVoxelWorldInitialState", payload)

    def init_memory(self):
        """Intialize the agent memory and logging."""
        low_level_data = self.low_level_data.copy()
        low_level_data["check_inside"] = heuristic_perception.check_inside

        self.memory = mc_memory.MCAgentMemory(
            db_file=os.environ.get("DB_FILE", ":memory:"),
            coordinate_transforms=self.coordinate_transforms,
            db_log_path="agent_memory.{}.log".format(self.name),
            agent_time=MCTime(self.get_world_time),
            agent_low_level_data=low_level_data,
        )
        # Add all dances to memory
        dance.add_default_dances(self.memory)
        file_log_handler = logging.FileHandler("agent.{}.log".format(self.name))
        file_log_handler.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_log_handler)

        # add method to convert coordinates from cuberite to droidlet:
        self.memory.to_droidlet_coords = from_minecraft_xyz_to_droidlet

        logging.info("Initialized agent memory")

    def init_perception(self):
        """Initialize perception modules"""
        self.perception_modules = {}
        self.perception_modules["language_understanding"] = NSPQuerier(self.opts, self)
        self.perception_modules["low_level"] = LowLevelMCPerception(self)
        self.perception_modules["heuristic"] = heuristic_perception.PerceptionWrapper(
            self,
            low_level_data=self.low_level_data,
            mark_airtouching_blocks=self.mark_airtouching_blocks,
        )

        # manual edits from dashboard
        self.perception_modules["dashboard"] = ManualChangesPerception(self)
        @sio.on("manual_change")
        def make_manual_change(sid, change):
            self.perception_modules["dashboard"].process_change(change)

        # set up the detection model
        # TODO: @kavya to check that this gets passed in when running the agent
        # TODO: fetch text_span here ?
        if self.opts.detection_model_path and os.path.isfile(self.opts.detection_model_path):
            self.perception_modules["detection_model"] = DetectionWrapper(
                model=self.opts.detection_model_path,
                agent=self
            )

    def init_controller(self):
        """Initialize all controllers"""
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
            "allow_clarification": self.allow_clarification,
        }
        self.dialogue_manager = DialogueManager(
            memory=self.memory,
            dialogue_object_classes=dialogue_object_classes,
            opts=self.opts,
            low_level_interpreter_data=low_level_interpreter_data,
        )

    def run_voxel_model(self, model, spans):
        rx, ry, rz = model.radius
        x, y, z = self.pos
        yzxb = self.get_blocks(x - rx, x + rx, y - ry, y + ry, z - rz, z + rz)
        blocks = np.ascontiguousarray(yzxb.transpose([2, 0, 1, 3]))
        model_out = model.perceive(blocks, text_spans=spans, offset=(x - rx, y - ry, z - rz))
        print(f"Voxel model run output: {model_out}")
        return model_out

    def perceive(self, force=False):
        """Whenever something is changed, that area is be put into a
        buffer which will be force-perceived by the agent in the next step.

        Here the agent first clusters areas that are overlapping on the buffer,
        then runs through all perception modules to perceive and finally clears the
        buffer when perception is done.

        The agent sends all perception updates to memory in order for them to
        update the memory state.
        """
        # 1. perceive from NLU parser
        ref_obj_spans = super().perceive()
        # 2. perceive from low_level perception module
        low_level_perception_output = self.perception_modules["low_level"].perceive()
        self.areas_to_perceive = cluster_areas(self.areas_to_perceive)
        self.areas_to_perceive = self.memory.update(
            low_level_perception_output, self.areas_to_perceive
        )["areas_to_perceive"]
        # 3. with the updated areas_to_perceive, perceive from heuristic perception module
        if force or not self.memory.task_stack_peek():
            # perceive from heuristic perception module
            heuristic_perception_output = self.perception_modules["heuristic"].perceive()
            self.memory.update(heuristic_perception_output)
        self.areas_to_perceive = []
        # 4. perceive any manual edits made from frontend
        dashboard_perception_output = self.perception_modules["dashboard"].perceive()
        self.memory.update(dashboard_perception_output)
        
        # 5. If detection model is initialized and text_span for reference object exists in
        # logical form, call perceive().
        if "detection_model" in self.perception_modules and ref_obj_spans:
            model = self.perception_modules["detection_model"]
            self.memory.update(self.run_voxel_model(model, ref_obj_spans))

        # 6. update dashboard world and map
        self.update_dashboard_world()
        
        @sio.on("toggle_map")
        def handle_toggle_map(sid, data):
            self.dash_enable_map = data["dash_enable_map"]
            #self.draw_map_to_dashboard()
        if self.opts.draw_map and self.dash_enable_map:
            if datetime.now() >= self.map_last_updated + timedelta(seconds=0.05*self.opts.map_update_ticks):
                self.map_last_updated = datetime.now()
                self.draw_map_to_dashboard()

    def get_time(self):
        """round to 100th of second, return as
        n hundreth of seconds since agent init.
        Returns:
            Current time in the world.
        """
        return self.memory.get_time()

    def get_world_time(self):
        """MC time is based on ticks, where 20 ticks happen every second.
        There are 24000 ticks in a day, making Minecraft days exactly 20 minutes long.
        The time of day in MC is based on the timestamp modulo 24000 (default).
        0 is sunrise, 6000 is noon, 12000 is sunset, and 18000 is midnight.

        Returns:
            Time of day based on above
        """
        return self.get_time_of_day()

    def safe_get_changed_blocks(self):
        """Get all blocks that have been changed.
        Returns:
            List of changed blocks
        """
        blocks = self.mover.get_changed_blocks()
        if isinstance(self.mover, PyWorldMover):
            blocks = [(xyz, (idm[0], idm[1])) for (xyz, idm) in blocks.items()]
        safe_blocks = deepcopy(blocks)
        if len(self.point_targets) > 0:
            for point_target in self.point_targets:
                pt = point_target[0]
                for b in blocks:
                    x, y, z = b[0]
                    xbad = x >= pt[0] and x <= pt[3]
                    ybad = y >= pt[1] and y <= pt[4]
                    zbad = z >= pt[2] and z <= pt[5]
                    if xbad and ybad and zbad:
                        if b in safe_blocks:
                            safe_blocks.remove(b)
        else:
            safe_blocks = blocks
        
        return safe_blocks

    def point_at(self, target, sleep=0):
        """Bot pointing.

        Args:
            target: list of x1 y1 z1 x2 y2 z2, where:
                    x1 <= x2,
                    y1 <= y2,
                    z1 <= z2.
        """
        assert len(target) == 6
        self.point_targets.append((target, time.time()))

        # TODO: put this in mover
        if self.backend == "cuberite":
            # flip x to move from droidlet coords to  cuberite coords
            target = [-target[3], target[1], target[2], -target[0], target[4], target[5]]

        point_json = build_question_json("/point {} {} {} {} {} {}".format(*target))
        self.send_chat(point_json)

        # sleep before the bot can take any actions
        # otherwise there might be bugs since the object is flashing
        # deal with this in the task...
        time.sleep(sleep)

    ###FIXME!!
    #    self.get_incoming_chats = self.get_chats

    # FIXME: normalize, switch in DSL to radians.
    def relative_head_pitch(self, angle):
        """Converts assistant's current pitch and yaw
        into a pitch and yaw relative to the angle."""
        new_pitch = self.get_player().look.pitch - angle
        self.set_look(self.get_player().look.yaw, new_pitch)

    def send_chat(self, chat: str):
        """Send chat from agent to player"""
        chat_json = False
        try:
            chat_json = json.loads(chat)
            chat_text = list(filter(lambda x: x["id"] == "text", chat_json["content"]))[0][
                "content"
            ]
        except:
            chat_text = chat

        logging.info("Sending chat: {}".format(chat_text))
        chat_memid = self.memory.nodes[ChatNode.NODE_TYPE].create(
            self.memory, self.memory.self_memid, chat_text
        )

        if chat_json and not isinstance(chat_json, int):
            chat_json["chat_memid"] = chat_memid
            chat_json["timestamp"] = round(datetime.timestamp(datetime.now()) * 1000)
            # Send the socket event to show this reply on dashboard
            sio.emit("showAssistantReply", chat_json)
        else:
            sio.emit("showAssistantReply", {"agent_reply": "Agent: {}".format(chat_text)})

        return self.mover.send_chat(chat_text)

    def get_detected_objects_for_map(self):
        search_res = self.memory.basic_search("SELECT MEMORY FROM ReferenceObject")
        mems = []
        if search_res is not None:
            _, mems = search_res
        detections_for_map = []
        for mem in mems:
            if hasattr(mem, "pos"):
                id_str = "no_id" if not hasattr(mem, "obj_id") else mem.obj_id
                obj = vars(mem)
                obj.pop('agent_memory', None)   # not necessary to show memory object type and location 
                obj["node_type"] = type(mem).__name__
                obj["obj_id"] = id_str
                obj["pos"] = list(mem.pos)
                detections_for_map.append(obj)
        return detections_for_map

    def draw_map_to_dashboard(self, obstacles=None, xyyaw=None):
        detections_for_map = []
        if not obstacles:
            obstacles = self.memory.place_field.get_obstacle_list()
            # if we are getting obstacles from memory, get detections from memory for map too
            detections_for_map = self.get_detected_objects_for_map()
        if not xyyaw:
            agent_pos = self.get_player().pos  # position of agent's feet
            agent_look = self.get_player().look
            mc_xyz = agent_pos.x, agent_pos.y, agent_pos.z
            mc_look = Look(agent_look.yaw, agent_look.pitch)
            x, _, z = from_minecraft_xyz_to_droidlet(mc_xyz)
            yaw, _ = from_minecraft_look_to_droidlet(mc_look)
            xyyaw = (x, z, yaw)
        triples = self.memory._db_read("SELECT * FROM Triples")
        sio.emit(
            "map",
            {
                "x": xyyaw[0],
                "y": xyyaw[1],
                "yaw": xyyaw[2],
                "map": obstacles,
                "bot_data": detections_for_map[0],
                "detections_from_memory": detections_for_map[1:],
                "triples": triples,
            },
        )

    def update_agent_pos_dashboard(self):
        agent_pos = self.get_player().pos
        payload = {
            "status": "updateVoxelWorldState",
            "world_state": {
                "agent": [
                    {
                        "name": "agent",
                        "x": float(agent_pos.x),
                        "y": float(agent_pos.y),
                        "z": float(agent_pos.z),
                    }
                ]
            },
        }
        sio.emit("updateVoxelWorldState", payload)

    def update_dashboard_world(self):
        MAX_RADIUS = 2
        agent_pos = self.get_player().pos
        x, y, z = round(agent_pos.x), round(agent_pos.y), round(agent_pos.z)
        origin = (x - MAX_RADIUS, y - MAX_RADIUS, z - MAX_RADIUS)
        yzxb = self.get_blocks(
            x - MAX_RADIUS,
            x + MAX_RADIUS,
            y - MAX_RADIUS,
            y + MAX_RADIUS,
            z - MAX_RADIUS,
            z + MAX_RADIUS,
        )

        # modified from util but keep air blocks
        def npy_to_blocks_list(npy, origin):
            import numpy as np

            blocks = []
            sy, sz, sx, _ = npy.shape
            for ry in range(sy):
                for rz in range(sz):
                    for rx in range(sx):
                        idm = tuple(npy[ry, rz, rx, :])
                        xyz = tuple(np.array([rx, ry, rz]) + origin)
                        blocks.append((xyz, idm))
            return blocks

        blocks = npy_to_blocks_list(yzxb, origin=origin)
        blocks = [
            ((int(xyz[0]), int(xyz[1]), int(xyz[2])), (int(idm[0]), int(idm[1])))
            for xyz, idm in blocks
        ]
        payload = {"status": "updateVoxelWorldState", "world_state": {"block": blocks}}
        sio.emit("updateVoxelWorldState", payload)

    def step_pos_x(self):
        self.mover.step_pos_x()
        self.update_agent_pos_dashboard()

    def step_neg_x(self):
        self.mover.step_neg_x()
        self.update_agent_pos_dashboard()

    def step_pos_y(self):
        self.mover.step_pos_y()
        self.update_agent_pos_dashboard()

    def step_neg_y(self):
        self.mover.step_neg_y()
        self.update_agent_pos_dashboard()

    def step_pos_z(self):
        self.mover.step_pos_z()
        self.update_agent_pos_dashboard()

    def step_neg_z(self):
        self.mover.step_neg_z()
        self.update_agent_pos_dashboard()

    def step_forward(self):
        self.mover.step_forward()
        self.update_agent_pos_dashboard()

    # TODO rename things a bit- some perceptual things are here,
    #      but under current abstraction should be in init_perception
    def init_physical_interfaces(self):
        """Initializes the physical interfaces of the agent."""
        # For testing agent without cuberite server
        if self.opts.port == -1:
            return
        if self.backend == "cuberite":
            from droidlet.lowlevel.minecraft.mc_agent import Agent as MCAgent

            logging.info(
                "Attempting to connect to cuberite cagent on port {}".format(self.opts.port)
            )
            self.cagent = MCAgent("localhost", self.opts.port, self.name)
            logging.info("Logged in to server")
            self.mover = CraftassistMover(self.cagent)
        elif self.backend == "pyworld":

            logging.info("Attempting to connect to pyworld on port {}".format(self.opts.port))
            # TODO allow pyworld ip to not be localhost
            try:
                self.mover = PyWorldMover(self.opts.port)
                self.cagent = None
                logging.info("Logged in to server")
            except:
                raise Exception("unable to connect to PyWorld on port {}".format(self.opts.port))
        else:
            raise Exception("unknown backend option {}".format(self.backend))

        for m in dir(self.mover):
            if callable(getattr(self.mover, m)) and m[0] != "_" and getattr(self, m, None) is None:
                setattr(self, m, getattr(self.mover, m))
        self.get_incoming_chats = self.get_chats
        self.get_other_players = self.get_all_players
        self.get_player_line_of_sight = self.get_all_player_line_of_sight

    def add_self_memory_node(self):
        """Adds agent node into its own memory"""
        # how/when to, memory is initialized before physical interfaces...
        try:
            p = self.get_player()
        except:  # this is for test/test_agent
            return
        SelfNode.update(self.memory, p, memid=self.memory.self_memid)


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Minecraft", base_path)
    opts = parser.parse()

    logging.basicConfig(level=opts.log_level.upper())

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))

    # Check that models and datasets are up to date and download latest resources.
    # Also fetches additional resources for internal users.
    if not opts.dev:
        try_download_artifacts(agent="craftassist")

    set_start_method("spawn", force=True)

    sa = CraftAssistAgent(opts)
    sa.start()
