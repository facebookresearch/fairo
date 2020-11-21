"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from base_log_reader import BaseLogReader

import hooks
import util


class PrintLogReader(BaseLogReader):
    def __init__(self, *args, ignore_hooks=[], only_hooks=[], **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            len(only_hooks) == 0 or len(ignore_hooks) == 0
        ), "Can't specify both only_hooks and ignore_hooks"

        for hid, hook_name in util.HOOK_MAP.items():
            if (len(ignore_hooks) > 0 and hid in ignore_hooks) or (
                len(only_hooks) > 0 and hid not in only_hooks
            ):
                continue
            func_name = "on_" + hook_name.lower()
            func = lambda *x, name=hook_name: print(*x[:2], name, *x[2:])
            setattr(self, func_name, func)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="Cuberite workdir; should contain settings.ini")
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--ignore", nargs="+", default=[])
    args = parser.parse_args()

    only_hooks = [getattr(hooks, h.upper()) for h in args.only]
    ignore_hooks = [getattr(hooks, h.upper()) for h in args.ignore]

    PrintLogReader(args.logdir, only_hooks=only_hooks, ignore_hooks=ignore_hooks).start()
