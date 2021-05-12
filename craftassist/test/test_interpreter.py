"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from unittest.mock import Mock

import craftassist.agent.heuristic_perception as heuristic_perception
import craftassist.agent.shapes as shapes
from droidlet.dialog.dialogue_objects import SPEAKERLOOK
from base_craftassist_test_case import BaseCraftassistTestCase
from droidlet.base_util import NextDialogueStep
from typing import List
from craftassist.agent.mc_util import Block, strip_idmeta, euclid_dist
from droidlet.interpreter.tests.all_test_commands import *


def add_two_cubes(test):
    triples = {"has_name": "cube", "has_shape": "cube"}
    test.cube_right: List[Block] = list(
        test.add_object(
            xyzbms=shapes.cube(bid=(42, 0)), origin=(9, 63, 4), relations=triples
        ).blocks.items()
    )
    test.cube_left: List[Block] = list(
        test.add_object(xyzbms=shapes.cube(), origin=(9, 63, 10), relations=triples).blocks.items()
    )
    test.set_looking_at(test.cube_right[0][0])


class TwoCubesInterpreterTest(BaseCraftassistTestCase):
    """A basic general-purpose test suite in a world which begins with two cubes.

    N.B. by default, the agent is looking at cube_right
    """

    def setUp(self):
        super().setUp()
        add_two_cubes(self)

    def test_noop(self):
        d = OTHER_COMMANDS["the weather is good"]
        changes = self.handle_logical_form(d)
        self.assertEqual(len(changes), 0)

    def test_destroy_that(self):
        d = DESTROY_COMMANDS["destroy where I am looking"]
        self.handle_logical_form(d)

        # Check that cube_right is destroyed
        self.assertEqual(
            set(self.get_idm_at_locs(strip_idmeta(self.cube_right)).values()), set([(0, 0)])
        )

    def test_copy_that(self):
        d = BUILD_COMMANDS["copy where I am looking to here"]
        changes = self.handle_logical_form(d)

        # check that another gold cube was built
        self.assert_schematics_equal(list(changes.items()), self.cube_right)

    def test_build_small_sphere(self):
        d = BUILD_COMMANDS["build a small sphere"]
        changes = self.handle_logical_form(d)

        # check that a small object was built
        self.assertGreater(len(changes), 0)
        self.assertLess(len(changes), 30)

    def test_build_1x1x1_cube(self):
        d = BUILD_COMMANDS["build a 1x1x1 cube"]
        changes = self.handle_logical_form(d)

        # check that a single block will be built
        self.assertEqual(len(changes), 1)

    def test_move_coordinates(self):
        d = MOVE_COMMANDS["move to -7 63 -8"]
        self.handle_logical_form(d)

        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, (-7, 63, -8)), 1)

    def test_move_here(self):
        d = MOVE_COMMANDS["move here"]
        self.handle_logical_form(d)

        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_build_diamond(self):
        d = BUILD_COMMANDS["build a diamond"]
        changes = self.handle_logical_form(d)

        # check that a Build was added with a single diamond block
        self.assertEqual(len(changes), 1)
        self.assertEqual(list(changes.values())[0], (57, 0))

    def test_build_gold_cube(self):
        d = BUILD_COMMANDS["build a gold cube"]
        changes = self.handle_logical_form(d)

        # check that a Build was added with a gold blocks
        self.assertGreater(len(changes), 0)
        self.assertEqual(set(changes.values()), set([(41, 0)]))

    def test_fill_all_holes_no_holes(self):
        d = FILL_COMMANDS["fill all holes where I am looking"]
        heuristic_perception.get_all_nearby_holes = Mock(return_value=[])  # no holes
        self.handle_logical_form(d)

    def test_go_to_the_tree(self):
        d = MOVE_COMMANDS["go to the tree"]
        try:
            self.handle_logical_form(d)
        except NextDialogueStep:
            pass

    def test_build_has_base(self):
        d = BUILD_COMMANDS["build a 9 x 9 stone rectangle"]
        self.handle_logical_form(d)

    def test_build_square_has_height(self):
        d = BUILD_COMMANDS["build a square with height 1"]
        changes = self.handle_logical_form(d)
        ys = set([y for (x, y, z) in changes.keys()])
        self.assertEqual(len(ys), 1)  # height 1

    def test_action_sequence_order(self):
        d = COMBINED_COMMANDS["move to 3 63 2 then 7 63 7"]
        self.handle_logical_form(d)
        self.assertLessEqual(euclid_dist(self.agent.pos, (7, 63, 7)), 1)

    def test_stop(self):
        # start moving
        target = (20, 63, 20)
        d = MOVE_COMMANDS["move to 20 63 20"]
        self.handle_logical_form(d, max_steps=5)

        # stop
        d = OTHER_COMMANDS["stop"]
        self.handle_logical_form(d)

        # assert that move did not complete
        self.assertGreater(euclid_dist(self.agent.pos, target), 1)

    def test_build_sphere_move_here(self):
        d = COMBINED_COMMANDS["build a small sphere then move here"]
        changes = self.handle_logical_form(d)

        # check that a small object was built
        self.assertGreater(len(changes), 0)
        self.assertLess(len(changes), 30)

        # check that agent moved
        self.assertLessEqual(euclid_dist(self.agent.pos, self.get_speaker_pos()), 1)

    def test_copy_that_and_build_cube(self):
        d = COMBINED_COMMANDS["copy where I am looking to here then build a 1x1x1 cube"]
        changes = self.handle_logical_form(d)

        # check that the cube_right is rebuilt and an additional block is built
        self.assertEqual(len(changes), len(self.cube_right) + 1)


class DigTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.set_looking_at((0, 63, 0))

    def test_dance(self):
        d = DIG_COMMANDS["dig a hole"]
        changes = self.handle_logical_form(d)
        # check agent changed a block:
        self.assertGreater(len(changes), 0)
        # check that all changes replaced blocks with air:
        assert not any([l[0] for l in list(changes.values())])


# doesn't actually check if the bot dances, just if it crashes FIXME!
# use recorder class from e2e_env
class DanceTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.set_looking_at((0, 63, 0))

    def test_dance(self):
        d = DANCE_COMMANDS["dance"]
        self.handle_logical_form(d)


@unittest.skip("these just check if the modifies run, not if they are accurate")
class ModifyTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.cube_right: List[Block] = list(
            self.add_object(shapes.cube(bid=(42, 0)), (9, 63, 4)).blocks.items()
        )
        self.set_looking_at(self.cube_right[0][0])

    def gen_modify(self, modify_dict):
        d = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action_sequence": [
                {
                    "action_type": "MODIFY",
                    "reference_object": {"filters": {"location": SPEAKERLOOK}},
                    "modify_dict": modify_dict,
                }
            ],
        }
        return d

    def test_modify(self):
        bigger = {"modify_type": "SCALE", "categorical_scale_factor": "BIGGER"}
        d = self.gen_modify(bigger)
        self.handle_logical_form(d)

        replace_bygeom = {
            "modify_type": "REPLACE",
            "new_block": {"has_colour": "green"},
            "replace_geometry": {"relative_direction": "LEFT", "amount": "QUARTER"},
        }
        d = self.gen_modify(replace_bygeom)
        self.handle_logical_form(d)

        shorter = {"modify_type": "SCALE", "categorical_scale_factor": "SHORTER"}
        d = self.gen_modify(shorter)
        self.handle_logical_form(d)

        # need to replace has_block_type by FILTERS....
        replace_byblock = {
            "modify_type": "REPLACE",
            "old_block": {"has_block_type": "iron"},
            "new_block": {"has_block_type": "diamond"},
        }
        d = self.gen_modify(replace_byblock)
        self.handle_logical_form(d)

        replace_bygeom = {
            "modify_type": "REPLACE",
            "new_block": {"has_block_type": "orange wool"},
            "replace_geometry": {"relative_direction": "LEFT", "amount": "QUARTER"},
        }
        d = self.gen_modify(replace_bygeom)
        self.handle_logical_form(d)


class SpawnSheep(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.set_looking_at((0, 63, 0))

    def test_spawn_5_sheep(self):
        d = SPAWN_COMMANDS["spawn 5 sheep"]
        self.handle_logical_form(d)
        self.assertEqual(len(self.agent.get_mobs()), 5)


# TODO ignore build outside of boundary in all tests
# TODO check its a circle
class CirclesLeftOfCircleTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        triples = {"has_name": "circle", "has_shape": "circle"}
        self.circle_right: List[Block] = list(
            self.add_object(
                xyzbms=shapes.circle(bid=(42, 0)), origin=(2, 63, 4), relations=triples
            ).blocks.items()
        )
        self.set_looking_at(self.circle_right[0][0])

    def test_build_other_circle(self):
        d = BUILD_COMMANDS["build a circle to the left of the circle"]
        changes = self.handle_logical_form(d)
        self.assertGreater(len(changes), 0)


class MoveBetweenTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        add_two_cubes(self)

    def test_between_cubes(self):
        d = MOVE_COMMANDS["go between the cubes"]
        self.handle_logical_form(d)
        assert heuristic_perception.check_between(
            [
                self.agent,
                [loc for loc, idm in self.cube_right],
                [loc for loc, idm in self.cube_left],
            ]
        )


# TODO class BuildInsideTest(BaseCraftassistTestCase):


class DestroyRedCubeTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.set_looking_at((0, 63, 0))

    def test_destroy_red_cube(self):
        d = BUILD_COMMANDS["build a red cube"]
        changes = self.handle_logical_form(d)
        self.assertGreater(len(changes), 0)  # TODO check its red
        # TODO also build a blue one
        # TODO test where fake player also builds one
        (loc, idm) = list(changes.items())[0]
        self.set_looking_at(loc)
        d = DESTROY_COMMANDS["destroy the red cube"]
        self.handle_logical_form(d)
        self.assertEqual((self.agent.world.blocks[:, :, :, 0] == idm[0]).sum(), 0)


class DestroyEverythingTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        add_two_cubes(self)

    def test_destroy_everything(self):
        d = DESTROY_COMMANDS["destroy everything"]
        self.handle_logical_form(d)
        self.assertEqual((self.agent.world.blocks[:, :, :, 0] == 42).sum(), 0)
        self.assertEqual((self.agent.world.blocks[:, :, :, 0] == 41).sum(), 0)


class FillTest(BaseCraftassistTestCase):
    def setUp(self):
        super().setUp()
        self.hole_poss = [(x, 62, z) for x in (8, 9) for z in (10, 11)]
        self.set_blocks([(pos, (0, 0)) for pos in self.hole_poss])
        self.set_looking_at(self.hole_poss[0])
        self.assertEqual(set(self.get_idm_at_locs(self.hole_poss).values()), set([(0, 0)]))

    def test_fill_that(self):
        d = FILL_COMMANDS["fill where I am looking"]
        self.handle_logical_form(d)

        # Make sure hole is filled
        self.assertEqual(set(self.get_idm_at_locs(self.hole_poss).values()), set([(3, 0)]))

    def test_fill_with_block_type(self):
        d = FILL_COMMANDS["fill where I am looking with gold"]
        self.handle_logical_form(d)

        # Make sure hole is filled with gold
        self.assertEqual(set(self.get_idm_at_locs(self.hole_poss).values()), set([(41, 0)]))


if __name__ == "__main__":
    unittest.main()
