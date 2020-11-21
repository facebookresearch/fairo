"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
import pickle
from world import build_coord_shifts

# TODO replay instantiates world, replays in world
class Recorder:
    def __init__(self, agent=None, filepath=None):
        assert agent or filepath
        self.agent = agent

        self.tape = {}
        if filepath:
            self.load_from_file(filepath)
        else:
            self.initial_blocks = agent.world.blocks.copy()
            self.coord_shift = agent.world.coord_shift

        self.current_blocks = self.initial_blocks.copy()
        self.replay_step = 0

        _, from_world_coords = build_coord_shifts(self.coord_shift)
        self.from_world_coords = from_world_coords

    def load_from_file(self, filepath):
        with open(filepath, "rb") as f:
            d = pickle.load(f)
        self.last_entry = d["last_entry"]
        self.tape = d["tape"]
        self.initial_blocks = d["initial_blocks"]
        self.coord_shift = d["coord_shift"]

    def save_to_file(self, filepath):
        d = {}
        d["last_entry"] = self.last_entry
        d["tape"] = self.tape
        d["initial_blocks"] = self.initial_blocks
        d["coord_shift"] = self.coord_shift
        with open(filepath, "wb") as f:
            pickle.dump(d, f)

    def maybe_add_entry(self):
        if self.tape.get(self.agent.count) is None:
            self.last_entry = self.agent.count
            self.tape[self.agent.count] = {}

    def record_action(self, a):
        self.maybe_add_entry()
        if self.tape[self.agent.count].get("actions") is None:
            self.tape[self.agent.count]["actions"] = []
        self.tape[self.agent.count]["actions"].append(a)

    def record_mobs(self):
        self.maybe_add_entry()
        self.tape[self.agent.count]["mobs"] = self.agent.get_mobs()

    def record_players(self):
        self.maybe_add_entry()
        player_list = self.agent.get_other_players()
        self.tape[self.agent.count]["players"] = []
        for player_struct in player_list:
            loc = self.agent.get_player_line_of_sight(player_struct)
            self.tape[self.agent.count]["players"].append((player_struct, loc))

    def record_agent(self):
        self.maybe_add_entry()
        self.tape[self.agent.count]["agent"] = self.agent.get_player()
        self.tape[self.agent.count]["logical_form"] = self.agent.logical_form

    def record_block_changes(self):
        d = self.agent.world.blocks - self.current_blocks
        if d.any():
            diff_idx = np.transpose(d.nonzero())
            self.maybe_add_entry()
            self.tape[self.agent.count]["block_changes"] = []
            for idx in diff_idx:
                loc = tuple(idx[:3])
                idm = tuple(self.agent.world.blocks[loc[0], loc[1], loc[2], :])
                self.current_blocks[loc[0], loc[1], loc[2], :] = idm
                self.tape[self.agent.count]["block_changes"].append((loc, idm))

    def record_world(self):
        self.record_mobs()
        self.record_agent()
        self.record_players()
        self.record_block_changes()

    def get_last_record(self):
        if self.tape.get(self.last_entry):
            return self.tape[self.last_entry]
        else:
            return {}

    def rewind(self):
        self.replay_step = 0
        self.current_blocks = self.initial_blocks.copy()

    def __iter__(self):
        return self

    def __next__(self):
        if self.replay_step > self.last_entry:
            raise StopIteration
        r = self.tape.get(self.replay_step, {})
        if r.get("blocks_changes"):
            for loc, idm in r["block_changes"]:
                self.current_blocks[loc[0], loc[1], loc[2], :] = idm
        self.replay_step += 1
        return {
            "step": self.replay_step,
            "logical_form": r.get("logical_form"),
            "blocks": self.current_blocks,
            "block_changes": r.get("block_changes"),
            "mobs": r.get("mobs"),
            "agent": r.get("agent"),
            "players": r.get("players"),
            "actions": r.get("actions"),
        }
