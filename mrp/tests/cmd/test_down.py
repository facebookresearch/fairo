import pytest

import mrp


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
    mrp.cmd.up(procs=["proc"], reset_logs=True)


@pytest.fixture
def hanging_proc_external():
    # Define process to hang.
    mrp.process(
        name="proc_ext",
        runtime=mrp.Host(run_command=["sleep", "999"]),
        env={"foo": "bar"},
    )

    # Run proc
    mrp.cmd.up(procs=["proc_ext"], reset_logs=True)

    # HACK Remove from defined processes
    mrp.process_def.defined_processes.pop("proc_ext")


def test_down(reset, hanging_proc):
    mrp.cmd.down(wait=False)

    # proc should be told to go down
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.DOWN

    mrp.cmd.wait()


def test_down_all(reset, hanging_proc, hanging_proc_external):
    mrp.cmd.down(wait=False)

    # proc should be told to go down, but not proc_ext
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.DOWN
    assert mrp.life_cycle.system_state().procs["proc_ext"].ask == mrp.life_cycle.Ask.UP

    mrp.cmd.down(all=True, wait=False)

    # proc should be told to go down
    assert (
        mrp.life_cycle.system_state().procs["proc_ext"].ask == mrp.life_cycle.Ask.DOWN
    )

    mrp.cmd.wait()


def test_down_proc(reset, hanging_proc):
    mrp.cmd.down("not_proc", wait=False)

    # proc should still be alive
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.UP

    mrp.cmd.down(procs=["proc"], wait=False)

    # proc should be told to go down
    assert mrp.life_cycle.system_state().procs["proc"].ask == mrp.life_cycle.Ask.DOWN

    mrp.cmd.wait()
