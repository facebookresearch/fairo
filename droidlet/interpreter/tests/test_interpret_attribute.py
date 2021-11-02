"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from copy import deepcopy
from droidlet.memory.sql_memory import AgentMemory
from droidlet.interpreter.interpreter import Interpreter
from droidlet.interpreter.interpret_attributes import (
    interpret_span_value,
    maybe_specific_mem,
    interpret_linear_extent,
    interpret_task_info,
)
from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.interpreter.interpret_attributes import AttributeInterpreter
from droidlet.interpreter.tests import all_test_commands
from droidlet.memory.memory_filters import MemorySearcher
from droidlet.memory.memory_nodes import PlayerNode, AttentionNode
from droidlet.base_util import Pos, Look, Player


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.interpreter = Interpreter("test_sp", "NULL", AgentMemory())
        self.interpreter.subinterpret["attribute"] = AttributeInterpreter()

    def test_interpret_span_value(self):
        action_dict = "79"
        self.assertEqual(
            interpret_span_value(self.interpreter, "test_sp", action_dict).value, 79.0
        )

        action_dict = all_test_commands.ATTRIBUTES["visit time"]
        target = deepcopy(all_test_commands.ATTRIBUTES["visit time"])
        self.assertEqual(
            interpret_span_value(self.interpreter, "test_sp", action_dict).value, target
        )

    def test_maybe_specific_mem(self):
        ad = all_test_commands.REFERENCE_OBJECTS["house"]
        mem, search_data = maybe_specific_mem(self.interpreter, "test_sp", ad)
        self.assertEqual(mem, None)
        self.assertEqual(search_data, "SELECT MEMORY FROM ReferenceObject WHERE ((has_tag=house))")

    def test_interpret_linear_extent(self):
        ad = all_test_commands.LINEAR_EXTENTS["distance from the house"]
        # Since house doesn't exist in memory, this should raise an error: I don't know what you're referring to
        self.assertRaises(
            ErrorWithResponse,
            interpret_linear_extent,
            **{"interpreter": self.interpreter, "speaker": "test_sp", "d": ad}
        )

    def test_interpret_task_info(self):
        ad = {"task_info": {"reference_object": {"attribute": "HEIGHT"}}}
        result = interpret_task_info(self.interpreter, "test_sp", ad)
        self.assertEqual(type(result).__name__, "AttributeSequence")
        self.assertEqual(type(result.attributes[0]).__name__, "TripleWalk")
        self.assertEqual(type(result.attributes[1]).__name__, "BBoxSize")
        self.assertEqual(len(result.attributes), 2)
        self.assertEqual(len(result.attributes[0].path), 1)


def dummy_specify_locations(interpreter, speaker, mems, steps, reldir):
    return mems[0].get_pos(), None


class BasicSearchWithAttributesTest(unittest.TestCase):
    def test_linear_extent_search(self):
        self.memory = AgentMemory()
        joe_eid = 10
        joe_memid = PlayerNode.create(
            self.memory, Player(joe_eid, "joe", Pos(1, 0, 1), Look(0, 0))
        )
        jane_memid = PlayerNode.create(self.memory, Player(11, "jane", Pos(-10, 0, 3), Look(0, 0)))
        jules_memid = PlayerNode.create(
            self.memory, Player(12, "jules", Pos(-1, 0, 2), Look(0, 0))
        )
        # FIXME shouldn't need to do this, interpreter should back off to something else
        # if there is no AttentionNode to use in filter_by_sublocation
        AttentionNode.create(self.memory, [1, 0, 1], joe_eid)
        self.interpreter = Interpreter("joe", "NULL", self.memory)
        self.interpreter.subinterpret["attribute"] = AttributeInterpreter()
        self.interpreter.subinterpret["specify_locations"] = dummy_specify_locations
        l = all_test_commands.LINEAR_EXTENTS["distance from joe"]
        l["frame"] = "joe"
        a = interpret_linear_extent(self.interpreter, "joe", l)
        m = MemorySearcher()
        comparator = {
            "input_left": {"value_extractor": {"attribute": a}},
            "input_right": {"value_extractor": 5},
            "comparison_type": "GREATER_THAN",
        }
        query_dict = {
            "output": "MEMORY",
            "memory_type": "ReferenceObject",
            "where_clause": {"AND": [comparator]},
        }
        memids, _ = m.search(self.memory, query=query_dict)


if __name__ == "__main__":
    unittest.main()
