"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest

from droidlet.lowlevel.minecraft.shapes import cube
from agents.craftassist.tests.base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.memory.dialogue_stack import DialogueStack
from droidlet.interpreter.craftassist import DummyInterpreter
import droidlet.interpreter.tests.all_test_commands


def add_many_objects(test):
    # loc, size, material, time (in world steps~secs)
    cube_data = [
        [(-5, 63, 4), 3, (57, 0), 1],
        [(10, 63, 9), 5, (41, 0), 5],
        [(5, 63, 4), 4, (41, 0), 10],
    ]
    cube_triples = {"has_name": "cube", "has_shape": "cube"}
    test.shapes = [
        list(
            test.agent.add_object_ff_time(
                cd[3], xyzbms=cube(size=cd[1], bid=cd[2]), origin=cd[0], relations=cube_triples
            ).blocks.items()
        )
        for cd in cube_data
    ]
    test.set_looking_at(test.shapes[0][0][0])


class FiltersTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        dummy_dialogue_stack = DialogueStack()
        self.dummy_interpreter = DummyInterpreter("SPEAKER", memory=self.agent.memory)

        add_many_objects(self)

    def test_first_and_last(self):
        f = droidlet.interpreter.tests.all_test_commands.FILTERS[
            "number of blocks in the first thing built"
        ]
        l = droidlet.interpreter.tests.all_test_commands.FILTERS[
            "number of blocks in the last thing built"
        ]
        DI = self.dummy_interpreter

        b = DI.subinterpret["filters"](DI, "SPEAKER", f)
        mems, vals = b()
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0], 27)

        b = DI.subinterpret["filters"](DI, "SPEAKER", l)
        mems, vals = b()
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0], 64)

    def test_farthest(self):
        two_f = droidlet.interpreter.tests.all_test_commands.FILTERS["two farthest cubes"]
        one_f = droidlet.interpreter.tests.all_test_commands.FILTERS["the farthest cube"]
        DI = self.dummy_interpreter

        memids = []
        b = DI.subinterpret["filters"](DI, "SPEAKER", one_f)
        mems, vals = b()
        self.assertEqual(len(mems), 1)
        assert vals[0] > 9.3
        b = DI.subinterpret["filters"](DI, "SPEAKER", two_f)
        mems, vals = b()
        assert vals[0] > 9.0
        assert vals[1] > 9.0
        self.assertEqual(len(mems), 2)

    def test_random(self):
        o = droidlet.interpreter.tests.all_test_commands.FILTERS["a random cube"]
        t = droidlet.interpreter.tests.all_test_commands.FILTERS["two random cubes"]

        DI = self.dummy_interpreter

        memids = []
        num_tries = 25
        b = DI.subinterpret["filters"](DI, "SPEAKER", o)
        for i in range(num_tries):
            mems, vals = b()
            self.assertEqual(len(mems), 1)
            memids.append(mems[0])

        assert not all([m == memids[0] for m in memids])

        b = DI.subinterpret["filters"](DI, "SPEAKER", t)
        mems, vals = b()
        self.assertEqual(len(mems), 2)


if __name__ == "__main__":
    unittest.main()
