from fbrp import util
from fbrp import registrar
import a0
import argparse
import random
import signal
import sys


@registrar.register_command("logs")
class logs_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--old", default=False, action="store_true")
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
            util.fail(f"No processes found to log")

        colors = [
            "\u001b[31m",  # "Red"
            "\u001b[32m",  # "Green"
            "\u001b[33m",  # "Yellow"
            "\u001b[34m",  # "Blue"
            "\u001b[35m",  # "Magenta"
            "\u001b[36m",  # "Cyan"
        ]
        random.shuffle(colors)
        reset_color = "\u001b[0m"

        width = max(len(name) for name in procs)

        log_listeners = []

        def make_listener(i, name, def_):
            prefix = f"{colors[i % len(colors)]}{name}" + " " * (width - len(name))
            msg_tmpl = f"{prefix} | {{msg}}{reset_color}"

            def callback(pkt):
                print(msg_tmpl.format(msg=pkt.payload))

            with util.common_env_context(def_):
                log_listeners.append(
                    a0.LogListener(
                        name,
                        a0.LogLevel.DBG,
                        a0.INIT_OLDEST if args.old else a0.INIT_AWAIT_NEW,
                        a0.ITER_NEXT,
                        callback,
                    )
                )

        for i, (name, def_) in enumerate(procs.items()):
            make_listener(i, name, def_)

        def onsignal(signum, frame):
            sys.exit(0)

        signal.signal(signal.SIGINT, onsignal)
        signal.signal(signal.SIGTERM, onsignal)
        signal.pause()
