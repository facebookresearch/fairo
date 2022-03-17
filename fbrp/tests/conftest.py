import os
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Pytest hook that ignores successful fork exits."""
    pid = os.getpid()
    outcome = yield

    if pid != os.getpid():
        try:
            outcome.get_result()
        except SystemExit as err:
            pytest.exit(err.code)
        pytest.exit(1)

    # Wait for children to complete.
    while True:
        try:
            os.wait()
        except ChildProcessError:
            break
