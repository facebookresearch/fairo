"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import numpy as np
from copy import deepcopy
from agents.craftassist.tests.recorder import Recorder
from agents.craftassist.tests.fake_agent import FakeAgent, FakePlayer
from droidlet.lowlevel.minecraft.pyworld.world import World
from droidlet.shared_data_structs import MockOpt
from droidlet.lowlevel.minecraft.pyworld.utils import Player, Pos, Item, Look
from droidlet.perception.craftassist import rotation
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import (
    build_shape_scene,
    SL,
    GROUND_DEPTH,
    H,
)


PLAYER_NAME = "SPEAKER"
CACHABLE_PERCEPTION = ["language_understanding"]
TTAD_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../../droidlet/artifacts/models/nlu/")
TTAD_BERT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../../droidlet/artifacts/datasets/annotated_data/"
)

# FIXME! USE CONSTRAINTS TO MAKE TRACTABLE TASK
def build_shape_scene_with_constraints(args, task_data):
    J = build_shape_scene(args)
    return J


def default_scenario(args, task_data):
    scenario = {}
    scenario["task"] = deepcopy(task_data)
    scenario["command_text"] = task_data["command_text"]

    J = build_shape_scene_with_constraints(args, task_data)
    scenario["task"]["scene"] = J
    world_spec = {}
    world_spec["coord_shift"] = J["offset"]

    def init_blocks(world):
        for b in J["schematic_for_cuberite"]:
            world.place_block(((b["x"], b["y"], b["z"]), (b["id"], b["meta"])))

    world_spec["ground_generator"] = init_blocks

    speaker_pos = np.add(J["avatarInfo"]["pos"], J["offset"]).tolist()
    speaker = FakePlayer(
        Player(42, PLAYER_NAME, Pos(*speaker_pos), Look(*J["avatarInfo"]["look"]), Item(0, 0)),
        active=False,
        opts=None,
    )
    world_spec["players"] = [speaker]
    world_spec["mobs"] = []
    world_spec["item_stacks"] = []
    agent_pos = np.add(J["agentInfo"]["pos"], J["offset"]).tolist()
    # FIXME normalize this interface
    world_spec["agent"] = {"pos": agent_pos}
    scenario["world_spec"] = world_spec
    world_opts = MockOpt()
    world_opts.sl = SL
    scenario["world_opts"] = world_opts
    agent_opts = MockOpt()
    agent_opts.nsp_models_dir = TTAD_MODEL_DIR
    agent_opts.nsp_data_dir = TTAD_BERT_DATA_DIR
    agent_opts.e2e_mode = True
    scenario["agent_opts"] = agent_opts
    scenario["max_execution_steps"] = 100
    verifier = task_data["verifier_gen"](scenario)
    scenario["verifier"] = verifier
    return scenario


class BaseE2ETest:
    def __init__(self, scenario):
        self.scenario = scenario
        self.results = []
        self.speaker = PLAYER_NAME
        self.max_steps = scenario["max_execution_steps"]
        self.cached_perception_models = {}

    def reset_world(self):
        for k, v in self.cached_perception_models.items():
            # FIXME make a method to do this
            v.agent = None
        self.world = World(scenario["world_opts"], scenario["world_spec"])
        self.agent = FakeAgent(
            self.world,
            opts=scenario["agent_opts"],
            prebuilt_perception=self.cached_perception_models,
        )
        for k in CACHABLE_PERCEPTION:
            self.cached_perception_models[k] = self.agent.perception_modules[k]
        self.agent.recorder = Recorder(agent=self.agent)

    def agent_execute_command(self, reset=True):
        self.reset_world()
        self.world.add_incoming_chat(self.scenario["command_text"], self.speaker)
        for i in range(self.max_steps):
            self.agent.step()
            if self.agent_should_stop():
                break
        self.results.append(self.scenario["verifier"](self.agent.recorder))

    def agent_should_stop(self):
        stop = False
        _, interpreter_mems = self.agent.memory.basic_search(
            "SELECT MEMORY FROM Interpreter WHERE finished = 0"
        )
        if len(interpreter_mems) == 0 and not self.agent.memory.task_stack_peek():
            stop = True

        # stuck waiting for answer?
        _, answer_task_mems = self.agent.memory.basic_search(
            "SELECT MEMORY FROM Task WHERE (action_name=awaitresponse AND prio>-1)"
        )
        if answer_task_mems and not any([m.finished for m in answer_task_mems]):
            stop = True
        # assume no clarifications rn
        stop_on_chat = True
        if stop_on_chat and self.agent.get_last_outgoing_chat():
            stop = True
        return stop


def gen_come_here_verifier(data):
    def f(recorder):
        ax, ay, az = recorder.tape[S.agent.recorder.last_entry]["agent"].pos
        px, py, pz = recorder.tape[S.agent.recorder.last_entry]["players"][0]["player_struct"].pos
        if abs(ax - px) + abs(ay - py) + abs(az - pz) < 4:
            return True
        else:
            return False

    return f


def gen_move_reldir_verifier(data):
    # assumes that the correct refobj is in the world, if there is a refobj.
    reldir = data["task"]["reldir"]
    num_steps = data["task"].get("num_steps")
    ref_obj_name = data["task"].get("ref_obj_name", "AGENT")
    frame = data["task"].get("frame", "SPEAKER")
    abs_reldir_vec = rotation.DIRECTIONS[reldir]
    if not ref_obj_name:
        ref_obj_name = "AGENT"
    ip_tol = data.get("ip_tol", 0.8)
    steps_tol = data.get("steps_tol", 1.1)
    S = data["task"]["scene"]

    def f(recorder):
        tape = recorder.tape
        if ref_obj_name == "AGENT":
            rpos_list = [np.array(tape[0]["agent"].pos)]
        elif ref_obj_name == "SPEAKER":
            rpos_list = [np.array(tape[0]["players"][0]["player_struct"].pos)]
        else:
            rpos_list = []
            for i in S["inst_seg_tags"]:
                if ref_obj_name.lower() in i["tags"]:
                    rpos_list.append(np.mean(i["locs"], axis=0))
        if frame == "SPEAKER":
            yaw = tape[0]["players"][0]["player_struct"].look.yaw
        else:  # frame == "AGENT"
            yaw = tape[0]["agent"].look.yaw
        dir_vec = rotation.transform(abs_reldir_vec, yaw, 0, inverted=True)
        last_pos = np.array(tape[recorder.last_entry]["agent"].pos)
        success = False
        # check each possible matching object; if agent moved approximately the right angle
        # and the right distance verify correct
        for p in rpos_list:
            d = last_pos - p
            dn = np.linalg.norm(d + 0.00001)
            d = d / dn
            ip = d @ dir_vec
            if ip > ip_tol:
                if num_steps:
                    if abs(dn - num_steps) < steps_tol:
                        success = True
                else:
                    success = True
        return success

    return f


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--SL", type=int, default=SL)
    parser.add_argument("--H", type=int, default=H)
    parser.add_argument("--GROUND_DEPTH", type=int, default=GROUND_DEPTH)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--MAX_NUM_GROUND_HOLES", type=int, default=0)
    parser.add_argument("--fence", action="store_true", default=False)
    parser.add_argument("--cuberite_x_offset", type=int, default=-SL // 2)
    parser.add_argument("--cuberite_y_offset", type=int, default=63 - GROUND_DEPTH)
    parser.add_argument("--cuberite_z_offset", type=int, default=-SL // 2)
    parser.add_argument("--iglu_scenes", default="")
    parser.add_argument("--save_data_path", default="")
    args = parser.parse_args()

    task_data = {
        "reldir": "LEFT",
        "command_text": "move left",
        "verifier_gen": gen_move_reldir_verifier,
    }
    scenario = default_scenario(args, task_data)

    S = BaseE2ETest(scenario)
    S.agent_execute_command()
