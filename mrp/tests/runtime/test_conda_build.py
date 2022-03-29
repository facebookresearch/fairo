import a0
import mrp
import mrp.process_def
import os
import shutil
import subprocess


def clear_process_list():
    mrp.process_def.defined_processes.clear()


def reset_state(*procs):
    clear_process_list()
    for proc in procs:
        shutil.rmtree(
            os.path.expanduser(f"~/.config/mrp/conda/mrp_{proc}"), ignore_errors=True
        )


def read_logs(topic):
    output = []
    r = a0.ReaderSync(
        a0.File(a0.env.topic_tmpl_log().format(topic=topic)), a0.INIT_OLDEST
    )
    while r.can_read():
        output.append(r.read().payload.decode())
    return output


def test_conda_nobuild():
    reset_state("proc")

    # Define process to run Python 3.7.11
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.7.11"],
            run_command=["python", "--version"],
        ),
    )

    # Run the proc and wait for it to complete.
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    assert read_logs("proc") == ["Python 3.7.11\n"]

    # Define process to run Python 3.8.8
    clear_process_list()
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            run_command=["python", "--version"],
        ),
    )

    # Run the proc without building.
    mrp.cmd.up("proc", reset_logs=True, build=False)
    mrp.cmd.wait("proc")

    assert read_logs("proc") == ["Python 3.7.11\n"]

    # Run the proc with building.
    clear_process_list()
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            run_command=["python", "--version"],
        ),
    )

    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    assert read_logs("proc") == ["Python 3.8.8\n"]


def test_conda_build_cache():
    reset_state("proc")

    test_counter_path = "/tmp/test_counter"
    test_counter_increment_command = [
        "bash",
        "-c",
        'bc <<< "1 + $(</tmp/test_counter)" > /tmp/test_counter',
    ]
    open(test_counter_path, "w").write("0")

    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.7.11"],
            setup_commands=[test_counter_increment_command],
            run_command=[],
        ),
    )

    # Run the proc and wait for it to complete.
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should have run and incremented the counter.
    assert open(test_counter_path).read() == "1\n"

    # Run the proc again with no changes.
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should NOT have run and the counter should be unchanged.
    assert open(test_counter_path).read() == "1\n"

    # Defining the same process shouldn't change anything.
    clear_process_list()
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.7.11"],
            setup_commands=[test_counter_increment_command],
            run_command=[],
        ),
    )
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should NOT have run and the counter should be unchanged.
    assert open(test_counter_path).read() == "1\n"

    # Changing the dependencies should cause a rebuild.
    clear_process_list()
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            setup_commands=[test_counter_increment_command],
            run_command=[],
        ),
    )
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should have run and the counter should be incremented.
    assert open(test_counter_path).read() == "2\n"

    clear_process_list()

    # Changing the setup command should cause a rebuild.
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            setup_commands=[["echo", "0"], test_counter_increment_command],
            run_command=[],
        ),
    )
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should have run and the counter should be incremented.
    assert open(test_counter_path).read() == "3\n"

    # Remake with identical setup.
    clear_process_list()
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            setup_commands=[["echo", "0"], test_counter_increment_command],
            run_command=[],
        ),
    )
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should NOT have run and the counter should be incremented.
    assert open(test_counter_path).read() == "3\n"

    # Update the conda history, poisoning the cache.
    subprocess.run(["conda", "install", "-n", "mrp_proc", "-y", "pycparser"])

    # Catch that the conda env has been updated.
    clear_process_list()
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            setup_commands=[["echo", "0"], test_counter_increment_command],
            run_command=[],
        ),
    )
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # The setup should have run and the counter should be incremented.
    assert open(test_counter_path).read() == "4\n"
