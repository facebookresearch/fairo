from fbrp import life_cycle
from fbrp import process_def
from fbrp import util
from fbrp.cmd import _autocomplete
import a0
import asyncio
import click
import contextlib
import json
import os
import sys
import threading
import traceback
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
            fringe.extend(process_def.defined_processes[proc_name].deps)
        except KeyError:
            raise ValueError(f"Unknown process: {proc_name}")
        all_proc_names.add(proc_name)
    return all_proc_names


def get_proc_names(proc_names, include_deps):
    if not proc_names:
        return process_def.defined_processes.keys()

    if include_deps:
        proc_names = transitive_closure(proc_names)
    unknown_proc_names = [
        proc_name
        for proc_name in proc_names
        if proc_name not in process_def.defined_processes
    ]
    if unknown_proc_names:
        raise ValueError(f"Unknown proc_names: {', '.join(unknown_proc_names)}")
    if not proc_names:
        raise ValueError(f"No proc_names found")
    return proc_names


def down_existing(names: typing.List[str], force: bool):
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

    if not force:
        raise RuntimeError(
            f"Conflicting processes already running: {', '.join(active_proc)}"
        )

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
        raise RuntimeError(f"Existing processes did not down in a timely manner.")


@click.command()
@click.argument("procs", nargs=-1, shell_complete=_autocomplete.defined_processes)
@click.option("-v/-q", "--verbose/--quiet", is_flag=True, default=True)
@click.option("--deps/--nodeps", is_flag=True, default=True)
@click.option("--build/--nobuild", is_flag=True, default=True)
@click.option("--cache/--nocache", is_flag=True, default=True)
@click.option("--run/--norun", is_flag=True, default=True)
@click.option("-f", "--force/--noforce", is_flag=True, default=False)
@click.option("--reset_logs", is_flag=True, default=False)
def cli(
    *cmd_procs,
    procs=[],
    verbose=True,
    deps=True,
    build=True,
    cache=True,
    run=True,
    force=False,
    reset_logs=False,
):
    # Support procs as *args when using cmd syntax.
    procs += cmd_procs

    names = get_proc_names(procs, deps)
    names = [name for name in names if process_def.defined_processes[name].runtime]
    if not names:
        raise ValueError(f"No processes found")

    down_existing(names, force)

    if reset_logs:
        for name in names:
            a0.File.remove(f"{name}.log.a0")

    if build:
        for name in names:
            proc_def = process_def.defined_processes[name]
            click.echo(f"building {name}...")
            proc_def.runtime._build(name, proc_def, cache, verbose)
            click.echo(f"built {name}\n")

    if run:
        for name in names:
            click.echo(f"running {name}...")
            life_cycle.set_ask(name, life_cycle.Ask.UP)
            life_cycle.set_state(name, life_cycle.State.STARTING)

            if os.fork() != 0:
                continue

            os.chdir("/")
            os.setsid()
            os.umask(0)

            if os.fork() != 0:
                sys.exit(0)

            proc_def = process_def.defined_processes[name]

            # Set up configuration.
            with util.common_env_context(proc_def):
                a0.Cfg(a0.env.topic()).write(json.dumps(proc_def.cfg))

                with open(f"/tmp/fbrp_{name}.log", "w", buffering=1) as logfile:
                    with contextlib.redirect_stdout(
                        logfile
                    ), contextlib.redirect_stderr(logfile):
                        click.echo(f"-- Process start time {a0.TimeWall.now()}")
                        life_cycle.set_launcher_running(name, True)
                        try:
                            asyncio.run(
                                proc_def.runtime._launcher(name, proc_def).run()
                            )
                        except BaseException as e:
                            click.echo(f"FATAL: {e}")
                            traceback.print_exc()
                        life_cycle.set_launcher_running(name, False)
                        sys.exit(0)
