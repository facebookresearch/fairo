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


if __name__ == "__main__":
    unittest.main()
