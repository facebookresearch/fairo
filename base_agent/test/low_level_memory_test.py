"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest
from base_agent.memory_nodes import PlayerNode
from base_agent.sql_memory import AgentMemory
from base_agent.base_util import Pos, Look, Player


class IncrementTime:
    def __init__(self):
        self.time = 0

    def get_time(self):
        return self.time

    def add_tick(self):
        self.time += 1


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.time = IncrementTime()
        self.memory = AgentMemory(agent_time=self.time)

    def test_triggers(self):
        joe_memid = PlayerNode.create(self.memory, Player(10, "joe", Pos(1, 0, 1), Look(0, 0)))
        joe_tag_memid = self.memory.tag(joe_memid, "joe")
        jane_memid = PlayerNode.create(self.memory, Player(11, "jane", Pos(-1, 0, 1), Look(0, 0)))

        joe_mems = self.memory.basic_search({"base_exact": {"name": "joe"}, "triples": []})
        jane_mems = self.memory.basic_search({"base_exact": {"name": "jane"}, "triples": []})
        assert len(joe_mems) == 1
        assert len(jane_mems) == 1

        joe_mems_from_tag = self.memory.basic_search(
            {"base_exact": {}, "triples": [{"obj_text": "joe", "pred_text": "has_tag"}]}
        )
        jane_mems_from_tag = self.memory.basic_search(
            {"base_exact": {}, "triples": [{"obj_text": "jane", "pred_text": "has_tag"}]}
        )
        assert len(joe_mems_from_tag) == 1
        assert len(jane_mems_from_tag) == 0

        self.time.add_tick()
        brother_of_memid = self.memory.add_triple(
            subj=joe_memid, pred_text="brother_of", obj=jane_memid
        )
        sister_of_memid = self.memory.add_triple(
            subj=jane_memid, pred_text="sister_of", obj=joe_memid
        )

        triples = self.memory.get_triples(subj=jane_memid, pred_text="sister_of")
        assert len(triples) == 1

        self.time.add_tick()
        self.memory.db_write("UPDATE ReferenceObjects SET x=? WHERE uuid=?", 2, joe_memid)
        cmd = "SELECT updated_time FROM Memories WHERE uuid=?"
        joe_t = self.memory._db_read(cmd, joe_memid)[0][0]
        jane_t = self.memory._db_read(cmd, jane_memid)[0][0]
        assert joe_t == 2
        assert jane_t == 0

        self.memory.forget(joe_memid)
        triples = self.memory.get_triples(subj=jane_memid, pred_text="sister_of")
        assert len(triples) == 0


if __name__ == "__main__":
    unittest.main()
