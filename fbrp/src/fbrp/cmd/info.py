from fbrp import life_cycle
from fbrp import process_def
import click
import json


@click.command()
def cli():
    click.secho("System State:", bold=True)
    click.echo(json.dumps(life_cycle.system_state().asdict(), indent="  "))

    click.secho("Defined Processes:", bold=True)
    click.echo(
        json.dumps(
            {
                name: proc_def.asdict()
                for name, proc_def in process_def.defined_processes.items()
            },
            indent="  ",
        )
    )
