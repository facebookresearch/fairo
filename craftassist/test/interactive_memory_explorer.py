"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import craftassist.agent.shapes as shapes
from base_craftassist_test_case import BaseCraftassistTestCase
from base_agent.dialogue_stack import DialogueStack
from craftassist.agent.dialogue_objects import DummyInterpreter


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
                cd[3],
                xyzbms=shapes.cube(size=cd[1], bid=cd[2]),
                origin=cd[0],
                relations=cube_triples,
            ).blocks.items()
        )
        for cd in cube_data
    ]
    test.set_looking_at(test.shapes[0][0][0])


class MemoryExplorer(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        dummy_dialogue_stack = DialogueStack(self.agent, self.agent.memory)
        self.dummy_interpreter = DummyInterpreter(
            "SPEAKER",
            agent=self.agent,
            memory=self.agent.memory,
            dialogue_stack=dummy_dialogue_stack,
        )

        add_many_objects(self)


if __name__ == "__main__":
    import all_test_commands  # noqa
    import memory_filters as mf  # noqa

    M = MemoryExplorer()
    M.setUp()
    a = M.agent
    m = M.agent.memory

#    M.dummy_interpreter.subinterpret["filters"](M.dummy_interpreter, "SPEAKER", all_test_commands.FILTERS["that cow"])
# s = M.dummy_interpreter.subinterpret["filters"](M.dummy_interpreter, "SPEAKER", all_test_commands.FILTERS["the first thing that was built"])
