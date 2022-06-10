import pytest


def pytest_addoption(parser):
    parser.addoption("--flag_m", action="store", default="Whether to load real NSP model")


@pytest.fixture()
def flag_m(request):
    setattr(request.cls, "flag_m", request.config.getoption("--flag_m"))