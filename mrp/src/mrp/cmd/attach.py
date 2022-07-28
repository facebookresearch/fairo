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
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)

    def on_pty_stdout(pkt):
        os.write(sys.stdout.fileno(), pkt.payload[: 4 * 1024])

    def on_pty_stderr(pkt):
        os.write(sys.stderr.fileno(), pkt.payload[: 4 * 1024])

    pub_in = a0.Publisher(f"mrp/{proc}/stdin")
    sub_out = a0.Subscriber(f"mrp/{proc}/stdout", on_pty_stdout) # noqa: F841
    sub_err = a0.Subscriber(f"mrp/{proc}/stderr", on_pty_stderr) # noqa: F841

    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        info = life_cycle.proc_info(proc)
        if info.state in [life_cycle.State.STOPPING, life_cycle.State.STOPPED]:
            break

        if r:
            o = sys.stdin.buffer.read()
            # ctrl+z
            if o == b"\x1a":
                break
            pub_in.pub(o)

    termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)
