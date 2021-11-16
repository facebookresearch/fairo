from fbrp import life_cycle
from fbrp import registrar
import argparse


@registrar.register_command("down")
class down_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        parser.add_argument("proc", action="append", nargs="*")

    @staticmethod
    def exec(args: argparse.Namespace):
        procs = life_cycle.system_state().procs.keys()

        given_proc_names = args.proc[0]
        if given_proc_names:
            procs = set(procs) & set(given_proc_names)

        for proc_name in procs:
            life_cycle.set_ask(proc_name, life_cycle.Ask.DOWN)
