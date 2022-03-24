from mrp import life_cycle
import click


@click.command()
def cli():
    state = life_cycle.system_state()
    if not state.procs:
        click.echo("No processes found.")
        return

    name_col_width = max(len(name) for name in state.procs)

    for name, info in state.procs.items():
        suffix = ""
        if info.state == life_cycle.State.STOPPED:
            suffix = f"(stopped code={info.return_code})"
        click.echo(" ".join([name, " " * (name_col_width - len(name)), suffix]))
