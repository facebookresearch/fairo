from fbrp import registrar
from fbrp.process import process
from fbrp.runtime.conda import Conda
from fbrp.runtime.docker import Docker
from fbrp.util import NoEscape
import argparse


def main():
    import fbrp.cmd.down
    import fbrp.cmd.list
    import fbrp.cmd.logs
    import fbrp.cmd.ps
    import fbrp.cmd.up
    import fbrp.cmd.wait

    parser = argparse.ArgumentParser(prog="fbrp")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "cmd"

    for cmd_name, cmd_cls in registrar.defined_commands.items():
        cmd_parser = subparsers.add_parser(cmd_name)
        cmd_cls.define_argparse(cmd_parser)
        cmd_parser.set_defaults(func=cmd_cls.exec)

    args = parser.parse_args()

    args.func(args)


__all__ = ["main", "process", "NoEscape", "Docker", "Conda"]
