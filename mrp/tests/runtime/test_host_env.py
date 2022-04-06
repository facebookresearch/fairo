import mrp
import mrp.process_def


def test_host_env():
    # Reset defined processes.
    mrp.process_def.defined_processes.clear()

    # Define process to echo environment variables.
    mrp.process(
        name="proc",
        runtime=mrp.Host(run_command=["env"]),
        env={"foo": "bar"},
    )

    # Run the proc and wait for it to complete.
    mrp.cmd.up("proc", reset_logs=True)
    mrp.cmd.wait("proc")

    # Capture the stdout from proc.
    # TODO(lshamis): Modify mrp.cmd.logs to support output capture for these kinds of tests.
    import a0

    output = []
    r = a0.ReaderSync(
        a0.File(a0.env.topic_tmpl_log().format(topic="proc")), a0.INIT_OLDEST
    )
    while r.can_read():
        output.append(r.read().payload.decode())

    # Parse the stdout as env key-values. Assume one per line, "=" separated.
    captured_env = {}
    for line in output:
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        # Remove newline at end of val.
        val = val.strip()
        captured_env[key] = val

    # Sanity check env values.
    assert captured_env["MRP_NAME"] == "proc"
    assert captured_env["A0_TOPIC"] == "proc"
    assert captured_env["PYTHONUNBUFFERED"] == "1"
    assert captured_env["foo"] == "bar"
