import mrp
from mrp.process_def import defined_processes, process
from mrp.runtime.conda import Conda
from mrp.runtime.docker import Docker
from mrp.runtime.host import Host
from mrp.util import NoEscape
from importlib.machinery import SourceFileLoader
import click
import inspect
import os
import sys


@click.group()
def cli():
    pass


def main(*args):
    try:
        cli(*args)
    except Exception as ex:
        click.echo(ex, err=True)
        sys.exit(1)
    sys.exit(0)


def import_msetup(path, processes=None):
    # Get path relative to caller.
    caller_path = inspect.stack()[1].filename
    caller_dir = os.path.dirname(caller_path)
    target_path = os.path.join(caller_dir, path)

    # Add default filename if provided path is a directory.
    if os.path.isdir(target_path):
        target_path = os.path.join(target_path, "msetup.py")

    # We hook mrp.process to filter processes within the import.

    # Save the real function for use within the hook
    # and to restore post-import.
    process_impl = mrp.process

    # Replacement implementation.
    def hook(name, *args, **kwargs):
        if processes is None or name in processes:
            return process_impl(name, *args, **kwargs)

    # Attach the hook.
    mrp.process = hook

    # Import.
    SourceFileLoader("msetup", target_path).load_module()

    # Undo the hook.
    mrp.process = process_impl


class cmd:
    pass


# Register all commands dynamically into the cli and cmd classes
# to allow execution directly without cli+argv.
#
# TODO(lshamis): Maybe do something similar for runtimes.
#
# For example:
#   mrp.cmd.up(procs=["foo"])
#
#   sys.argv = ["foo"]
#   mrp.main()

this_file_path = os.path.dirname(os.path.realpath(__file__))
cmds_path = os.path.join(this_file_path, "cmd")
for cmd_file in os.listdir(cmds_path):
    if not cmd_file.startswith("_") and cmd_file.endswith(".py"):
        cmd_path = os.path.join(cmds_path, cmd_file)
        cmd_name = cmd_file[: -len(".py")]
        module = SourceFileLoader(cmd_name, cmd_path).load_module()

        setattr(cmd, cmd_name, module.cli.callback)
        cli.add_command(module.cli, cmd_name)

__all__ = [
    # CLI entrypoint.
    "main",
    # Process definition.
    "process",
    "defined_processes",
    "NoEscape",
    # Runtimes.
    "Docker",
    "Conda",
    "Host",
    # Command interface.
    "cmd",
]
