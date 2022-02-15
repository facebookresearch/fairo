from fbrp.cmd._common import CommonFlags
from fbrp.process import process
from fbrp.runtime.conda import Conda
from fbrp.runtime.docker import Docker
from fbrp.util import NoEscape
from importlib.machinery import SourceFileLoader
import click
import os


@click.group()
@click.option("-v/-q", "--verbose/--quiet", is_flag=True, default=False)
def cli(verbose):
    CommonFlags.verbose = verbose


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
