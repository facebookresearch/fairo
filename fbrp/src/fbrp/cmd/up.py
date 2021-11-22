from fbrp import life_cycle
from fbrp import registrar
from fbrp import util
from fbrp.cmd.base import BaseCommand
import a0
import argparse
import asyncio
import json
import os
import sys
import threading
import types
import typing


def transitive_closure(proc_names):
    all_proc_names = set()
    fringe = list(proc_names)
    while fringe:
        proc_name = fringe.pop()
        if proc_name in all_proc_names:
            continue
        try:
            fringe.extend(registrar.defined_processes[proc_name].deps)
        except KeyError:
            util.fail(f"Unknown process: {proc_name}")
        all_proc_names.add(proc_name)
    return all_proc_names


def get_proc_names(proc_names, include_deps):
    if not proc_names:
        return registrar.defined_processes.keys()

    if include_deps:
        proc_names = transitive_closure(proc_names)
    unknown_proc_names = [
        proc_name
        for proc_name in proc_names
        if proc_name not in registrar.defined_processes
    ]
    if unknown_proc_names:
        util.fail(f"Unknown proc_names: {', '.join(unknown_proc_names)}")
    if not proc_names:
        util.fail(f"No proc_names found")
    return proc_names


def down_existing(args: argparse.Namespace, names: typing.List[str]):
    def find_active_proc(system_state):
        return [
            name
            for name in names
            if name in system_state.procs
            and system_state.procs[name].state != life_cycle.State.STOPPED
        ]

    active_proc = find_active_proc(life_cycle.system_state())
    if not active_proc:
        return

    if not args.force:
        util.fail(f"Conflicting processes already running: {', '.join(active_proc)}")

    for name in active_proc:
        life_cycle.set_ask(name, life_cycle.Ask.DOWN)

    ns = types.SimpleNamespace()
    ns.cv = threading.Condition()
    ns.sat = False

    def callback(system_state):
        if not find_active_proc(system_state):
            with ns.cv:
                ns.sat = True
                ns.cv.notify()

    watcher = life_cycle.system_state_watcher(callback)

    with ns.cv:
        success = ns.cv.wait_for(lambda: ns.sat, timeout=3.0)

    if not success:
        util.fail(f"Existing processes did not down in a timely manner.")


@registrar.register_command("up")
class up_cmd(BaseCommand):
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--deps", default=True, action="store_true")
        parser.add_argument("--nodeps", dest="deps", action="store_false")
        parser.add_argument("--build", default=True, action="store_true")
        parser.add_argument("--nobuild", dest="build", action="store_false")
        parser.add_argument("--run", default=True, action="store_true")
        parser.add_argument("--norun", dest="run", action="store_false")
        parser.add_argument("-f", "--force", default=False, action="store_true")
        parser.add_argument("proc", action="append", nargs="*")

    @staticmethod
    def exec(args: argparse.Namespace):
        names = get_proc_names(args.proc[0], args.deps)
        names = [name for name in names if registrar.defined_processes[name].runtime]
        if not names:
            util.fail(f"No processes found")

        down_existing(args, names)

        if args.build:
            for name in names:
                proc_def = registrar.defined_processes[name]
                print(f"building {name}...")
                proc_def.runtime._build(name, proc_def, args)
                print(f"built {name}\n")

        if args.run:
            for name in names:
                print(f"running {name}...")
                life_cycle.set_ask(name, life_cycle.Ask.UP)

                if os.fork() != 0:
                    continue

                os.chdir("/")
                os.setsid()
                os.umask(0)

                if os.fork() != 0:
                    sys.exit(0)

                proc_def = registrar.defined_processes[name]

                # Set up configuration.
                with util.common_env_context(proc_def):
                    a0.Cfg(a0.env.topic()).write(json.dumps(proc_def.cfg))
                    life_cycle.set_launcher_running(name, True)
                    asyncio.run(proc_def.runtime._launcher(name, proc_def, args).run())
                    life_cycle.set_launcher_running(name, False)
                    sys.exit(0)
