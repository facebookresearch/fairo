"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from test.fake_agent import FakeAgent

from typing import Dict
from mc_util import Player, Pos, Look, Item, XYZ, IDM
from world import World, Opt, flat_ground_generator


class BaseSQLMockEnvironment:
    def __init__(self):
        """Replica of test environment"""
        spec = {
            "players": [Player(42, "SPEAKER", Pos(5, 63, 5), Look(270, 0), Item(0, 0))],
            "mobs": [],
            "ground_generator": flat_ground_generator,
            "agent": {"pos": (0, 63, 0)},
            "coord_shift": (-16, 54, -16),
        }
        world_opts = Opt()
        world_opts.sl = 32
        self.world = World(world_opts, spec)
        self.agent = FakeAgent(self.world, opts=None)
        self.speaker = "cat"

    def handle_logical_form(self, logical_form: Dict, chatstr: str = "") -> Dict[XYZ, IDM]:
        """Handle an action dict and call self.flush()"""
        obj = self.agent.dialogue_manager.handle_logical_form(self.speaker, logical_form, chatstr)
        if obj is not None:
            self.agent.dialogue_manager.dialogue_stack.append(obj)
        self.flush()
        return obj

    def flush(self) -> Dict[XYZ, IDM]:
        """Update memory and step the dialogue stacks"""
        self.agent.dialogue_manager.dialogue_stack.step()
        return
