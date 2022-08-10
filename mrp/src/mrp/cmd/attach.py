from mrp import life_cycle
from mrp.cmd import _autocomplete
import a0
import click
import sys
import os
import tty
import termios
import select
import fcntl


@click.command()
@click.argument("proc", type=str, shell_complete=_autocomplete.running_processes)
def cli(proc):
    # Set the user's terminal to raw mode.
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    # Get file access mode for stdin.
    orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
    # Edit the stdin flag setting to be non-blocking.
    # This modifies the behavior of sys.stdin.buffer.read() to
    # return all available data, rather than wait for a line break
    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)

    def make_writer(fd):
        # Write directly to the fd.
        # This avoids buffering and the flush required by sys.stdout.buffer.write.
        return lambda pkt: os.write(fd, pkt.payload[: 4 * 1024])

    pub_in = a0.Publisher(f"mrp/{proc}/stdin")
    sub_out = a0.Subscriber(  # noqa: F841
        f"mrp/{proc}/stdout", make_writer(sys.stdout.fileno())
    )
    sub_err = a0.Subscriber(  # noqa: F841
        f"mrp/{proc}/stderr", make_writer(sys.stderr.fileno())
    )

    while True:
        # Wait until stdin is available, or a timeout of 0.1s.
        r, _, _ = select.select([sys.stdin], [], [], 0.1)

        # Check if the process is still running.
        info = life_cycle.proc_info(proc)
        if info.state in [life_cycle.State.STOPPING, life_cycle.State.STOPPED]:
            break

        # Check if stdin is available.
        if r:
            data = sys.stdin.buffer.read()
            # ctrl+z
            if data == b"\x1a":
                break
            pub_in.pub(data)

    # Reset the user's terminal to the mode pre-attach.
    termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)
