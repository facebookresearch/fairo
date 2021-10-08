from facebook_robotics_platform.setup import registrar
from facebook_robotics_platform.setup.process import process
from facebook_robotics_platform.setup.runtime.conda import Conda
from facebook_robotics_platform.setup.runtime.docker import Docker
import argparse


def main():
    import facebook_robotics_platform.setup.cmd.down
    import facebook_robotics_platform.setup.cmd.list
    import facebook_robotics_platform.setup.cmd.logs
    import facebook_robotics_platform.setup.cmd.up

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


__all__ = ["main", "process", "Docker", "Conda"]
