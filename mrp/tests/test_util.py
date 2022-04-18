from mrp import util


def test_empty():
    assert util.shell_join([]) == ""


def test_basic():
    assert util.shell_join(["a", "b", "c"]) == "a b c"


def test_escape():
    assert util.shell_join(["a", ";b", "c"]) == "a ';b' c"


def test_noescape():
    assert util.shell_join(["a", util.NoEscape(";b"), "c"]) == "a ;b c"
