from fbrp.process_def import process
from fbrp.runtime.conda import Conda
from fbrp.runtime.docker import Docker
from fbrp.util import NoEscape
from importlib.machinery import SourceFileLoader
import click
import os


@click.group()
def cli():
    pass


def main():
    cmd_list = [
        "down",
        "info",
        "logs",
        "ps",
        "up",
        "wait",
    ]

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    for cmd in cmd_list:
        path = os.path.join(this_file_path, f"cmd/{cmd}.py")
        module = SourceFileLoader(cmd, path).load_module()
        cli.add_command(module.cli, cmd)

    cli()


__all__ = ["main", "process", "NoEscape", "Docker", "Conda"]
