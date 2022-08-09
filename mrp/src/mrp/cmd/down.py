import mrp
from mrp import life_cycle
from mrp import process_def
from mrp.cmd import _autocomplete
import click


@click.command()
@click.option("--local", is_flag=True)
@click.option("--wait/--nowait", is_flag=True)
@click.argument("procs", nargs=-1, shell_complete=_autocomplete.running_processes)
def cli(*cmd_procs, procs=None, local=False, wait=True):
    procs = procs or []
    procs += cmd_procs

    specified_procs = set(procs)

    # Get all MRP procs running in the system.
    running_procs = set(
        name
        for name, info in life_cycle.system_state().procs.items()
        if info.state != life_cycle.State.STOPPED
    )

    # Get MRP procs defined in the local msetup.py
    defined_procs = set(process_def.defined_processes.keys())

    if local:  # Down local msetup.py
        assert not procs, "Cannot use '--local' flag and also specify processes."
        down_procs = running_procs & defined_procs

    elif procs:  # Specified process set.
        down_procs = running_procs & specified_procs

    else:  # System-wide down.
        down_procs = running_procs

    # Ask all the procs to go down.
    for proc in down_procs:
        click.echo(f"stopping {proc}...")
        life_cycle.set_ask(proc, life_cycle.Ask.DOWN)

    # Wait for them to finished shutting down.
    if wait and down_procs:
        mrp.cmd.wait(*down_procs)
