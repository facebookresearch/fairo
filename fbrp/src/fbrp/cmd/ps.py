from fbrp import life_cycle
from fbrp import registrar
import a0
import argparse
import json


@registrar.register_command("ps")
class ps_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        pass

    @staticmethod
    def exec(args: argparse.Namespace):
        state = life_cycle.system_state()
        if not state.procs:
            print("No processes found.")
            return

        name_col_width = max(len(name) for name in state.procs)

        for name, info in state.procs.items():
            suffix = ""
            if info.state == life_cycle.State.STOPPED:
                suffix = f"(stopped code={info.return_code})"
            print(name, " " * (name_col_width - len(name)), suffix)
