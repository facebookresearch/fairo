"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from world import World, SimpleMob, make_mob_opts, Opt
from utils import Player, Pos, Look, Item
from fake_agent import FakeAgent
from world_visualizer import Window, setup
from recorder import Recorder
import pyglet
import logging

if __name__ == "__main__":
    log_formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
    )
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().handlers.clear()
    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(log_formatter)
    logging.getLogger().addHandler(sh)

    opts = Opt()
    opts.sl = 32
    spec = {
        "players": [Player(42, "SPEAKER", Pos(0, 68, 0), Look(270, 80), Item(0, 0))],
        "mobs": [SimpleMob(make_mob_opts("cow")), SimpleMob(make_mob_opts("chicken"))],
        "agent": {"pos": (1, 68, 1)},
        "coord_shift": (-opts.sl // 2, 63 - opts.sl // 2, -opts.sl // 2),
    }
    world = World(opts, spec)
    agent = FakeAgent(world, opts=None)
    speaker_name = agent.get_other_players()[0].name
    move_speaker_pos = {"action_type": "MOVE", "location": {"location_type": "SPEAKER_POS"}}
    build_small_sphere = {
        "action_type": "BUILD",
        "schematic": {"has_name": "sphere", "has_size": "small"},
    }

    build_medium_sphere = {
        "action_type": "BUILD",
        "schematic": {"has_name": "sphere", "has_size": "medium"},
    }
    build_small_sphere_here = {
        "action_type": "BUILD",
        "schematic": {"has_name": "sphere", "has_size": "small"},
        "location": {"location_type": "SPEAKER_POS"},
    }
    lf = {"dialogue_type": "HUMAN_GIVE_COMMAND", "action": build_medium_sphere}
    #    lf = {
    #            "dialogue_type": "HUMAN_GIVE_COMMAND",
    #            "action": move_speaker_pos,
    #        }
    dummy_chat = "TEST {}".format(lf)
    agent.set_logical_form(lf, dummy_chat, speaker_name)

    agent.recorder = Recorder(agent=agent)
    for i in range(100):
        agent.step()
    FNAME = "test_record.pkl"
    agent.recorder.save_to_file(FNAME)
    new_recorder = Recorder(filepath=FNAME)
    W = Window(recorder=new_recorder)
    #
    #    W = Window(agent.recorder, agent=agent)
    ##    # Hide the mouse cursor and prevent the mouse from leaving the window.
    ##    W.set_exclusive_mouse(True)
    setup()
    pyglet.app.run()
