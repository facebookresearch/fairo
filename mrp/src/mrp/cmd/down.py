import mrp
from mrp import life_cycle
from mrp import process_def
from mrp.cmd import _autocomplete
import click


@click.command()
@click.option("--all", is_flag=True)
@click.option("--wait/--nowait", is_flag=True)
@click.argument("procs", nargs=-1, shell_complete=_autocomplete.defined_processes)
def cli(*cmd_procs, procs=None, all=False, wait=True):
    procs = procs or []

    # Get all MRP procs running in the system
    running_procs = life_cycle.system_state().procs.keys()
    down_procs = set(running_procs)

    if all:  # system-wide down
        assert not procs, "Specifying processes is not supported with the flag '--all'."
        assert not cmd_procs, "Specifying processes is not supported when all=True."

    else:  #  local down (only processes defined within the current msetup.py)
        defined_procs = process_def.defined_processes.keys()
        down_procs = down_procs & set(defined_procs)

        # Support procs as *args when using cmd syntax.
        procs += cmd_procs
        if procs:
            down_procs = down_procs & set(procs)

    for proc in down_procs:
        click.echo(f"stopping {proc}...")
        life_cycle.set_ask(proc, life_cycle.Ask.DOWN)

    if wait and down_procs:
        mrp.cmd.wait(*down_procs)
