"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import unittest

import droidlet.lowlevel.minecraft.size_words as size_words


class TestSizeWords(unittest.TestCase):
    def assert_in_range(self, x, rng):
        a, b = rng
        self.assertTrue(a <= x < b)

    def test_str_to_int(self):
        x = size_words.size_str_to_int("big")
        self.assert_in_range(x, size_words.RANGES["large"])

    def test_str_to_int_mod(self):
        x = size_words.size_str_to_int("really big")
        self.assert_in_range(x, size_words.RANGES["huge"])


if __name__ == "__main__":
    unittest.main()
