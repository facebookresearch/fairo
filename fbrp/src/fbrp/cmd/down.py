from fbrp import util
from fbrp import registrar
import a0
import argparse
import json


@registrar.register_command("down")
class down_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        parser.add_argument("proc", action="append", nargs="*")

    @staticmethod
    def exec(args: argparse.Namespace):
        procs = {
            name: def_
            for name, def_ in registrar.defined_processes.items()
            if def_.runtime
        }

        given_proc_names = args.proc[0]
        if given_proc_names:
            procs = {
                name: def_ for name, def_ in procs.items() if name in given_proc_names
            }

        if not procs:
            util.fail(f"No processes found to down")

        for name, def_ in procs.items():
            with util.common_env_context(def_):
                a0.Publisher(f"fbrp/control/{name}").pub(
                    json.dumps(
                        dict(
                            action="down",
                        )
                    )
                )
