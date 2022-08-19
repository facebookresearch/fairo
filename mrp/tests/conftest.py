import mrp


def pytest_runtest_setup(item):
    mrp.defined_processes.clear()
