from fbrp.process import ProcDef
import typing

defined_commands: typing.Dict[str, "BaseCommand"] = {}
defined_processes: typing.Dict[str, ProcDef] = {}


def register_command(name):
    def wrap(cls):
        defined_commands[name] = cls
        return cls

    return wrap
