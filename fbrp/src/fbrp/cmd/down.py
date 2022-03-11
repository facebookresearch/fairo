from fbrp import life_cycle
from fbrp.cmd import _autocomplete
import click


@click.command()
@click.argument("procs", nargs=-1, shell_complete=_autocomplete.running_processes)
def cli(*cmd_procs, procs=[]):
    # Support procs as *args when using cmd syntax.
    procs += cmd_procs

    down_procs = life_cycle.system_state().procs.keys()

    if procs:
        down_procs = set(down_procs) & set(procs)

    for proc in down_procs:
        life_cycle.set_ask(proc, life_cycle.Ask.DOWN)
