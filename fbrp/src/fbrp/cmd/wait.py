from fbrp import life_cycle
from fbrp.cmd import _autocomplete
import click
import threading
import types


@click.command()
@click.argument("procs", nargs=-1, shell_complete=_autocomplete.running_processes)
@click.option("-t", "--timeout", type=float, default=0)
def cli(*cmd_procs, procs=[], timeout=0):
    # Support procs as *args when using cmd syntax.
    procs += cmd_procs

    wait_procs = set(life_cycle.system_state().procs.keys())
    if procs:
        wait_procs = set(wait_procs) & set(procs)

    ns = types.SimpleNamespace(
        cv=threading.Condition(),
        sat=False,
    )

    def callback(sys_state):
        for proc, info in sys_state.procs.items():
            if proc in wait_procs and info.state != life_cycle.State.STOPPED:
                return
        with ns.cv:
            ns.sat = True
            ns.cv.notify()

    watcher = life_cycle.system_state_watcher(callback)

    with ns.cv:
        ns.cv.wait_for(lambda: ns.sat, timeout=timeout or None)
