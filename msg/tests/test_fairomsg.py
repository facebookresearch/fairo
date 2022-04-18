import pytest
from fairomsg import get_msgs, get_pkgs

@pytest.fixture
def pkg_names():
    return get_pkgs()

def test_get_pkgs(pkg_names):
    assert len(pkg_names) > 0

def test_get_msgs(pkg_names):
    for pkg_name in pkg_names:
        get_msgs(pkg_name)