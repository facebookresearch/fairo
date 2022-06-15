import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--flag_load_nsp_model",
        action="store",
        default="False",
        help="Whether to load real NSP model",
    )


@pytest.fixture()
def flag_load_nsp_model(request):
    setattr(request.cls, "flag_load_nsp_model", request.config.getoption("--flag_load_nsp_model"))
