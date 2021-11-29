from fbrp import util
import unittest

class TestShellJoin(unittest.TestCase):
    def test_empty(self):
        assert util.shell_join([]) == ""

    def test_basic(self):
        assert util.shell_join(["a", "b", "c"]) == "a b c"

    def test_escape(self):
        assert util.shell_join(["a", ";b", "c"]) == "a ';b' c"

    def test_noescape(self):
        assert util.shell_join(["a", util.NoEscape(";b"), "c"]) == "a ;b c"
