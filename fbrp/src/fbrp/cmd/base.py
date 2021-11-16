import argparse


class BaseCommand:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        raise NotImplementedError("Command hasn't implemented define_argparse!")

    @staticmethod
    def exec(args: argparse.Namespace):
        raise NotImplementedError("Command hasn't implemented exec!")
