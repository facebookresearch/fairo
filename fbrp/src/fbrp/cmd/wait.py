from fbrp import life_cycle
import click
import threading
import types


@click.command()
@click.argument("procs", nargs=-1)
@click.option("-t", "--timeout", type=float, default=0)
def cli(procs, timeout):
    wait_procs = set(life_cycle.system_state().procs.keys())
    if procs:
        wait_procs = set(wait_procs) & set(procs)

    ns = types.SimpleNamespace()
    ns.cv = threading.Condition()
    ns.sat = False

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
