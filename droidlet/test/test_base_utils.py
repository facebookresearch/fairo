"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import hashlib
import numpy as np
from droidlet.base_util import *

class BasicTest(unittest.TestCase):
    def test_number_from_span(self):
        assert number_from_span("eighteen") == 18.0
        assert number_from_span("20") == 20.0

    def test_hash_username(self):
        username = "kavya"
        salt = uuid.uuid4().hex
        hashed_user = hashlib.sha256(salt.encode() + username.encode()).hexdigest() + ":" + salt
        assert check_username(hashed_user, username) == True
        assert check_username(hash_user(username), username) == True

    def test_get_bounds(self):
        locations = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
        assert get_bounds(locations) == (0, 4, 1, 5, 2, 6)

    def test_group_by(self):
        def even_num_detector(x):
            if x % 2:
                return "False"
            return "True"
        items = [1, 2, 3, 4, 5, 6]
        out_dict = group_by(items, even_num_detector)
        expected_dict = {"True": [2, 4, 6], "False": [1, 3, 5]}
        for k in out_dict.keys():
            assert k in expected_dict
            assert out_dict[k] == expected_dict[k]

    def test_distance_computation(self):
        point1 = [1, 2, 3]
        point2 = [10, 2, 4]
        # test euclid_dist
        assert np.floor(euclid_dist(point1, point2)) == 9.0
        # test manhat_dist
        assert np.floor(manhat_dist(point1, point2)) == 10.0

    def test_pos_to_np(self):
        new_pos = Pos(10, 10, 23)
        assert type(pos_to_np(new_pos)) == np.ndarray

    def test_to_player_struct(self):
        pos = [0, 0, 0]
        yaw = 0.0
        pitch = 2.0
        eid = 10
        name = "dadada"
        player = to_player_struct(pos, yaw, pitch, eid, name)
        assert player.name == "dadada"
        assert type(player.pos) == Pos
        assert type(player.look) == Look
        assert player.pos.x == 0
        assert player.pos.y == 0
        assert player.pos.z == 0
        assert player.look.yaw == 0.0
        assert player.look.pitch == 2.0

    def test_npy_to_blocks_list(self):
        test_array = np.array([[[[190, 0], [190, 0]],
                                [[190, 0], [0, 0]],
                                [[190, 0], [0, 0]],
                                [[190, 0], [0, 0]]]])
        assert type(test_array) == np.ndarray
        output = npy_to_blocks_list(test_array)
        assert type(output) == list
        assert len(output) == 5
        assert output[0][1] == (190, 0)

    def test_cube(self):
        cube_struct = cube()
        assert len(cube_struct) == 27   # number of blocks
        cube_struct = cube(size=2)
        assert len(cube_struct) == 8
        cube_struct = cube(size=4, bid=(1, 1))
        assert len(cube_struct) == 64
        assert cube_struct[10][1] == (1, 1) # check block id, meta
        assert cube_struct[0][0] == (0, 0, 0)


if __name__ == "__main__":
    unittest.main()
