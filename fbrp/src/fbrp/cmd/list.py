from fbrp import registrar
import argparse


@registrar.register_command("list")
class list_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        parser.add_argument("proc", action="append", nargs="*")

    @staticmethod
    def exec(args: argparse.Namespace):
        print("Defined Processes:")
        for proc in registrar.defined_processes:
            print(f"  {proc}")
