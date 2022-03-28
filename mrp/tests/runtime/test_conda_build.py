import a0
import mrp
import mrp.process_def

def read_logs(topic):
    output = []
    r = a0.ReaderSync(
        a0.File(a0.env.topic_tmpl_log().format(topic=topic)), a0.INIT_OLDEST
    )
    while r.can_read():
        output.append(r.read().payload.decode())
    return output


def test_conda_build():
    # Reset defined processes.
    mrp.process_def.defined_processes.clear()

    # Define process to run Python 3.7.11
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.7.11"],
            run_command=["python3", "--version"],
        ),
    )

    # Run the proc and wait for it to complete.
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    assert read_logs("proc") == ["Python 3.7.11\n"]

    # Reset defined processes.
    mrp.process_def.defined_processes.clear()

    # Define process to run Python 3.8.8
    mrp.process(
        name="proc",
        runtime=mrp.Conda(
            dependencies=["python=3.8.8"],
            run_command=["python3", "--version"],
        ),
    )

    # Run the proc without building.
    mrp.cmd.up("proc", reset_logs=True, build=False)
    mrp.cmd.wait("proc")

    assert read_logs("proc") == ["Python 3.7.11\n"]

    # Run the proc with building.
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    assert read_logs("proc") == ["Python 3.8.8\n"]
