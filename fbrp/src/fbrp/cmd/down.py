from fbrp import life_cycle
import click


@click.command()
@click.argument("procs", nargs=-1)
def cli(procs):
    down_procs = life_cycle.system_state().procs.keys()

    if procs:
        down_procs = set(down_procs) & set(procs)

    for proc in down_procs:
        life_cycle.set_ask(proc, life_cycle.Ask.DOWN)
