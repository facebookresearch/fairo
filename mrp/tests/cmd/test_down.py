import pytest

import mrp


@pytest.fixture
def hanging_proc():
    # Reset defined processes.
    mrp.process_def.defined_processes.clear()

    # Define process to hang.
    mrp.process(
        name="proc",
        runtime=mrp.Host(run_command=["sleep", "999"]),
        env={"foo": "bar"},
    )

    # Run proc
    mrp.cmd.up("proc", reset_logs=True)


def test_down(hanging_proc):
    mrp.cmd.down()
    mrp.cmd.wait()

    # Proc should be told to go down
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.DOWN


def test_down_all(hanging_proc):
    mrp.cmd.down(all=True)
    mrp.cmd.wait()

    # Proc should be told to go down
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.DOWN


def test_down_proc(hanging_proc):
    mrp.cmd.down("not_proc")
    mrp.cmd.wait("not_proc")

    # Proc should still be alive
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.UP

    mrp.cmd.down(procs=["proc"])
    mrp.cmd.wait()

    # Proc should be told to go down
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.DOWN
