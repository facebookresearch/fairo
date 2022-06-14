import pytest


def pytest_addoption(parser):
    parser.addoption("--flag_LoadNspModel", action="store_true", help="Whether to load real NSP model")


@pytest.fixture()
def flag_LoadNspModel(request):
    setattr(request.cls, "flag_LoadNspModel", request.config.getoption("--flag_LoadNspModel"))
