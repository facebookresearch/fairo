from fbrp import util
from fbrp.process import defined_processes
import a0
import click
import random
import signal
import sys


@click.command()
@click.argument("procs", nargs=-1)
@click.option("-o", "--old", is_flag=True, default=False)
def cli(procs, old):
    # Find all defined processes.
    display_procs = defined_processes.items()
    # Filter out processes that have no runtime defined.
    # These processes were meant to chain or combine other processes, but haven't
    # gotten much use yet. Do we want to keep them?
    display_procs = {name: def_ for name, def_ in display_procs if def_.runtime}

    # If processes have been specified, filter out the ones that aren't requested.
    if procs:
        display_procs = {
            name: def_ for name, def_ in display_procs.items() if name in procs
        }

    # Fail if no processes are left.
    if not display_procs:
        util.fail(f"No processes found to log")

    # Give each process a random color.
    colors = [
        "\u001b[31m",  # "Red"
        "\u001b[32m",  # "Green"
        "\u001b[33m",  # "Yellow"
        "\u001b[34m",  # "Blue"
        "\u001b[35m",  # "Magenta"
        "\u001b[36m",  # "Cyan"
    ]
    random.shuffle(colors)
    reset_color = "\u001b[0m"

    # There will be a left hand column with the process name.
    # Find a common width for the name column.
    width = max(len(name) for name in display_procs)

    log_listeners = []

    def make_listener(i, name, def_):
        # Cache the left hand column.
        prefix = f"{colors[i % len(colors)]}{name}" + " " * (width - len(name))
        msg_tmpl = f"{prefix} | {{msg}}{reset_color}"

        # On message received, print it to stdout.
        def callback(pkt):
            print(msg_tmpl.format(msg=pkt.payload))

        # Create the listener.
        with util.common_env_context(def_):
            log_listeners.append(
                a0.LogListener(
                    name,
                    # TODO(lshamis): Make a flag for log level.
                    a0.LogLevel.DBG,
                    a0.INIT_OLDEST if old else a0.INIT_AWAIT_NEW,
                    a0.ITER_NEXT,
                    callback,
                )
            )

    # Make a log listener for each process.
    for i, (name, def_) in enumerate(display_procs.items()):
        make_listener(i, name, def_)

    # Block until ctrl-c is pressed.
    def onsignal(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, onsignal)
    signal.signal(signal.SIGTERM, onsignal)
    signal.pause()
