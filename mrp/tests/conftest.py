import os
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Pytest hook that ignores forks."""
    pid = os.getpid()
    yield

    if pid != os.getpid():
        pytest.exit(reason="child process completed")

    # Wait for children to complete.
    while True:
        try:
            os.wait()
        except ChildProcessError:
            break
