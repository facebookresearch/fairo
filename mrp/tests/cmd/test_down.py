import pytest

import mrp
from mrp.life_cycle import State, Ask


@pytest.fixture
def reset():
    # Reset defined processes.
    mrp.process_def.defined_processes.clear()


@pytest.fixture
def hanging_proc():
    # Define process to hang.
    mrp.process(
        name="proc",
        runtime=mrp.Host(run_command=["sleep", "999"]),
        env={"foo": "bar"},
    )

    # Run proc
    mrp.cmd.up("proc", reset_logs=True)


@pytest.fixture
def hanging_proc_external():
    # Define process to hang.
    mrp.process(
        name="proc_ext",
        runtime=mrp.Host(run_command=["sleep", "999"]),
        env={"foo": "bar"},
    )

    # Run proc
    mrp.cmd.up("proc_ext", reset_logs=True)

    # HACK Remove from defined processes
    mrp.process_def.defined_processes.pop("proc_ext")


def test_down(reset, hanging_proc):
    # proc should be told to go down
    mrp.cmd.down()
    assert mrp.life_cycle.system_state().procs["proc"].ask == Ask.DOWN


def test_down_all(reset, hanging_proc, hanging_proc_external):
    # proc should be told to go down, but not proc_ext
    mrp.cmd.down(local=True)
    assert mrp.life_cycle.system_state().procs["proc"].ask == Ask.DOWN
    assert mrp.life_cycle.system_state().procs["proc_ext"].ask == Ask.UP

    # proc should be told to go down
    mrp.cmd.down()
    assert mrp.life_cycle.system_state().procs["proc_ext"].ask == Ask.DOWN


def test_down_proc(reset, hanging_proc):
    # proc should still be alive
    mrp.cmd.down("not_proc")
    assert mrp.life_cycle.system_state().procs["proc"].ask == Ask.UP

    # proc should be told to go down
    mrp.cmd.down("proc")
    assert mrp.life_cycle.system_state().procs["proc"].ask == Ask.DOWN


def test_wait(reset, hanging_proc):
    # proc should be told to go down, but down should return immediately with wait=False
    mrp.cmd.down("proc", wait=False)
    assert mrp.life_cycle.system_state().procs["proc"].ask == Ask.DOWN
    assert (
        mrp.life_cycle.system_state().procs["proc"].state == State.STARTING
        or State.STARTED
    )

    # proc should be stopped when expicit wait returns
    mrp.cmd.wait("proc")
    assert mrp.life_cycle.system_state().procs["proc"].state == State.STOPPED
