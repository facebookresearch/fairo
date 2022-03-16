from fbrp import util
from fbrp.cmd import _autocomplete
from fbrp import process_def
import a0
import click
import random
import signal
import sys


@click.command()
@click.argument(
    "procs",
    nargs=-1,
    # Logs connects to live feeds unless --old is set.
    # This autocomplete checks for the old flag.
    #   If --old is set, suggest any defined process that has logs.
    #   Otherwise, suggest running processes.
    shell_complete=_autocomplete.conditional(
        lambda ctx, unused_param, unused_incomplete: ctx.params["old"],
        _autocomplete.intersection(
            _autocomplete.alephzero_topics(protocol="log"),
            _autocomplete.defined_processes,
        ),
        _autocomplete.running_processes,
    ),
)
@click.option("-o", "--old", is_flag=True, default=False)
def cli(*cmd_procs, procs=[], old=False):
    # Support procs as *args when using cmd syntax.
    procs += cmd_procs

    # Find all defined processes.
    display_procs = process_def.defined_processes.items()
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
        raise ValueError(f"No processes found to log")

    # Give each process a random color.
    colors = [
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
    ]
    random.shuffle(colors)

    # There will be a left hand column with the process name.
    # Find a common width for the name column.
    width = max(len(name) for name in display_procs)

    log_listeners = []

    def make_listener(i, name, def_):
        # Cache the left hand column.
        prefix = f"{name}" + " " * (width - len(name))
        msg_tmpl = f"{prefix} | {{msg}}"

        # On message received, print it to stdout.
        def callback(pkt):
            click.secho(msg_tmpl.format(msg=pkt.payload), fg=colors[i % len(colors)])

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
