from facebook_robotics_platform.common import util
from facebook_robotics_platform.setup import registrar
from facebook_robotics_platform.setup.cmd.base import BaseCommand
import a0
import argparse
import asyncio
import os
import sys
import json


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


def get_proc_names(root_proc_names, include_deps):
    proc_names = root_proc_names or ["main"]
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

        if args.build:
            for name in names:
                proc_def = registrar.defined_processes[name]
                print(f"building {name}...")
                proc_def.runtime._build(name, proc_def, args)
                print(f"built {name}\n")

        if args.run:
            try:
                os.mkdir("/dev/shm/fbrp")
            except:
                pass

            for name in names:
                print(f"running {name}...")

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
                    asyncio.run(proc_def.runtime._launcher(name, proc_def, args).run())
                    sys.exit(0)
