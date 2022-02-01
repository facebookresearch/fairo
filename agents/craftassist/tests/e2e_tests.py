"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os
import numpy as np
from agents.craftassist.tests.recorder import Recorder
from agents.craftassist.tests.fake_agent import FakeAgent, FakePlayer
from droidlet.lowlevel.minecraft.pyworld.world import World
from droidlet.shared_data_structs import MockOpt
from droidlet.lowlevel.minecraft.pyworld.utils import (
    Player,
    Pos,
    Look,
    Item,
    Look,
    to_relative_pos,
)
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


def default_scenario(args, command_text, verifier):
    scenario = {}
    scenario["command_text"] = command_text

    J = build_shape_scene(args)
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
        world_spec = self.scenario["world_spec"]
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--SL", type=int, default=SL)
    parser.add_argument("--H", type=int, default=H)
    parser.add_argument("--GROUND_DEPTH", type=int, default=GROUND_DEPTH)
    parser.add_argument("--MAX_NUM_SHAPES", type=int, default=3)
    parser.add_argument("--NUM_SCENES", type=int, default=3)
    parser.add_argument("--fence", action="store_true", default=False)
    parser.add_argument("--cuberite_x_offset", type=int, default=-SL // 2)
    parser.add_argument("--cuberite_y_offset", type=int, default=63 - GROUND_DEPTH)
    parser.add_argument("--cuberite_z_offset", type=int, default=-SL // 2)
    parser.add_argument("--save_data_path", default="")
    args = parser.parse_args()

    def come_here_verifier(recorder):
        ax, ay, az = S.agent.recorder.tape[S.agent.recorder.last_entry]["agent"].pos
        px, py, pz = S.agent.recorder.tape[S.agent.recorder.last_entry]["players"][0][
            "player_struct"
        ].pos
        if abs(ax - px) + abs(ay - py) + abs(az - pz) < 4:
            return True
        else:
            return False

    scenario = default_scenario(args, "come_here", come_here_verifier)

    S = BaseE2ETest(scenario)
    S.agent_execute_command()
