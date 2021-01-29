"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np

# TODO rest of proprioception
class SelfPerception:
    def __init__(self, agent, perceive_freq=10):
        self.agent = agent
        self.memory = agent.memory
        self.perceive_freq = perceive_freq
        self.add_self_memory_node()

    def add_self_memory_node(self):
        # clean this up!  FIXME!!!!! put in base_agent_memory?
        # how/when to, memory is initialized before physical interfaces...
        # currently use self_memid for eid
        self.memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, name, ref_type, x, y, z, pitch, yaw) VALUES (?,?,?,?,?,?,?,?,?)",
            self.memory.self_memid,
            self.memory.self_memid,
            self.agent.name,
            "player",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def perceive(self, force=False):
        # FIXME (low pri) remove these in code, get from sql
        # TODO async path so we can ask for these every step
        if self.agent.count % self.perceive_freq == 0 or force:
            base_pos = self.agent.mover.get_base_pos_in_canonical_coords()
            pan = self.agent.mover.get_pan()
            tilt = self.agent.mover.get_tilt()
            # FIXME! agent pose memory type+data structure
            self.agent.pos = np.array(base_pos[:2], dtype="float32")
            self.agent.base_yaw = float(base_pos[2])
            # TODO get rid of this:
            self.agent.yaw = float(base_pos[2] + pan)
            self.agent.pan = float(pan)
            self.agent.pitch = float(tilt)
            self.agent.base_yaw = float(base_pos[2])
            self.update_self_memory()

    def update_self_memory(self):
        memid = self.agent.memory.self_memid
        cmd = "UPDATE ReferenceObjects SET eid=?, name=?, x=?,  y=?, z=?, pitch=?, yaw=? WHERE "
        cmd = cmd + "uuid=?"
        # using self_memid as eid too
        self.memory.db_write(
            cmd,
            memid,
            self.agent.name,
            self.agent.pos[0],
            0.0,
            self.agent.pos[1],
            self.agent.pitch,
            self.agent.yaw,
            memid,
        )
