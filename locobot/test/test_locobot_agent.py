"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import os
import Pyro4
import time
import logging
import sys
from filelock import FileLock
from locobot.test.utils import get_fake_detection
from locobot.agent.objects import AttributeDict
from locobot.agent.locobot_agent import LocobotAgent
from test_utils import assert_turn_degree

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(BASE_AGENT_ROOT)

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
IP = "127.0.0.1"
if os.getenv("LOCOBOT_IP"):
    IP = os.getenv("LOCOBOT_IP")

opts = AttributeDict(
    {
        "nsp_data_dir": "datasets/annotated_data/",
        "nsp_models_dir": "models/semantic_parser/",
        "ground_truth_data_dir": "datasets/ground_truth/",
        "no_ground_truth": True,
        "perception_model_dir": "models/perception",
        "incoming_chat_path": "incoming_chat.txt",
        "ip": IP,
        "backend": "habitat",
        "no_default_behavior": True,
    }
)


def get_locobot_agent():
    def fix_path(opts):
        agent_path = os.path.join(BASE_AGENT_ROOT, "locobot/agent")
        for optname, optval in opts.items():
            if "path" in optname or "dir" in optname:
                if optval:
                    opts[optname] = os.path.join(agent_path, optval)

    fix_path(opts)
    logging.info("INIT test_locobot_agent")
    return LocobotAgent(opts)


def set_agent_command(command):
    """Writes the command to opts.incoming_chat_path, which is here the agent
    reads from in every step."""
    with FileLock("chat.lock"):
        with open(opts.incoming_chat_path, "w") as f:
            f.write(str(int(time.time())) + " | <commandline_speaker> " + command)


class LocobotTests(unittest.TestCase):
    agent = get_locobot_agent()

    def setUp(self):
        # handpicked to be in front of empty space in current test scene apartment_0
        # initial_state = [[4.4, 2.08, -3]]
        # initial_state = [[4.8, 0.16, -1.0]]
        initial_state = [[0.0, 0.0, 0.0]]
        self.agent.mover.move_absolute(initial_state)

    def test_go_to_the_x(self):
        d = get_fake_detection("vase", ["a", "b", "c"], [-0.4, -0.08, 0.0])
        d.save_to_memory(self.agent)
        t = self.agent.memory.get_detected_objects_tagged("vase")
        logging.info("Created marker {}".format(t))
        set_agent_command("go to the vase")
        # TODO: Move coordinates are imprecise. Add assert after fixing that.
        self.agent.step()

    def test_turn(self):
        turn_commands = {
            "turn left 90 degrees": 90,
            "turn right 90 degrees": -90,
            "turn 90 degrees": -90,  # default to right turn
            "turn left 45 degrees": 45,
            "turn right 40 degrees": -40,
        }
        for command, degree in turn_commands.items():
            set_agent_command(command)
            init = self.agent.mover.bot.get_base_state(state_type="odom")
            self.agent.step()
            final = self.agent.mover.bot.get_base_state(state_type="odom")
            logging.info("initial state {}, final state {}".format(init, final))
            assert_turn_degree(init[2], final[2], degree)

    def test_move_relative(self):
        set_agent_command("go left 1 meter")

    #     TODO: Add assert after fixing the command
    # initial_state = self.agent.mover.bot.get_base_state(state_type="odom")
    # self.agent.step()
    # final_state = self.agent.mover.bot.get_base_state(state_type="odom")
    # distance = norm(array(initial_state)[:2] - array(final_state)[:2])
    # assert_allclose(distance, 1)

    def test_look_at_x(self):
        # look up test case: (pan, tilt) = (0.0, -0.46)
        self.agent.mover.look_at([0, 1.5, 1], 0.0, 0.0)
        time.sleep(10)
        pan, tilt = self.agent.mover.get_pan(), self.agent.mover.get_tilt()

        self.assertTrue(-0.49 <= tilt <= -0.43, "tilt is wrong after looking up at [0, 1.5, 1]")
        ##################################################################
        # look down test case: (pan, tilt) = (0.0, 0.46)
        self.agent.mover.look_at([0, 0.5, 1], 0.0, 0.0)
        time.sleep(10)
        # FIXME: command_finished() always return True
        # while not self.agent.mover.bot.command_finished():
        #     pass
        pan, tilt = self.agent.mover.get_pan(), self.agent.mover.get_tilt()

        self.assertTrue(0.43 <= tilt <= 0.49, "tilt is wrong after looking down at [0, 0.5, 1]")
        ##################################################################
        # look left test case: (pan, tilt) = (0.78, 0.0)
        self.agent.mover.look_at([-1, 1, 1], 0.0, 0.0)
        time.sleep(10)
        pan, tilt = self.agent.mover.get_pan(), self.agent.mover.get_tilt()

        self.assertTrue(0.75 <= pan <= 0.81, "pan is wrong after looking left at [-1, 1, 1]")
        ##################################################################
        # look left test case: (pan, tilt) = (-0.78, 0.0)
        self.agent.mover.look_at([1, 1, 1], 0.0, 0.0)
        time.sleep(10)
        pan, tilt = self.agent.mover.get_pan(), self.agent.mover.get_tilt()

        self.assertTrue(-0.81 <= pan <= -0.75, "pan is wrong after looking right at [1, 1, 1]")

    def test_point_at_x(self):
        # TBD: blocked until habitat has an arm for the locobot
        pass


if __name__ == "__main__":
    unittest.main()
