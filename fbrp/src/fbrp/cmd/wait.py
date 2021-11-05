from gc import callbacks
import a0
from fbrp import life_cycle
from fbrp import registrar
import argparse
import threading


@registrar.register_command("wait")
class wait_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        parser.add_argument("proc", action="append", nargs="*")

    @staticmethod
    def exec(args: argparse.Namespace):
        procs = life_cycle.system_state().procs.keys()

        given_proc_names = args.proc[0]
        if given_proc_names:
            procs = set(procs) & set(given_proc_names)

        class Namespace:
            pass

        ns = Namespace()
        ns.cv = threading.Condition()
        ns.sat = False

        def callback(sys_state):
            for proc, info in sys_state.procs.items():
                if proc in procs and info.state != life_cycle.State.STOPPED:
                    return
            with ns.cv:
                ns.sat = True
                ns.cv.notify()

        watcher = life_cycle.system_state_watcher(callback)

        with ns.cv:
            ns.cv.wait_for(lambda: ns.sat)
