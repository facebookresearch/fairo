from fbrp import life_cycle
from fbrp import util
from fbrp.process_def import ProcDef
from fbrp.runtime.base import BaseLauncher, BaseRuntime
import asyncio
import os
import pathlib
import signal
import subprocess
import typing


class Launcher(BaseLauncher):
    def __init__(
        self,
        run_command: typing.List[str],
        name: str,
        proc_def: ProcDef,
    ):
        self.run_command = run_command
        self.name = name
        self.proc_def = proc_def

    async def gather_cmd_outputs(self):
        async def log_pipe(logger, pipe):
            while True:
                try:
                    async for line in pipe:
                        logger(line)
                except ValueError:
                    # TODO(lshamis): Can we grab the line in chucks?
                    logger("<[FBRP] line length exceeded. skipping>")
                    continue
                else:
                    break

        self.down_task = asyncio.create_task(self.down_watcher(self.handle_down))
        try:
            await asyncio.gather(
                log_pipe(util.stdout_logger(), self.proc.stdout),
                log_pipe(util.stderr_logger(), self.proc.stderr),
                self.log_psutil(),
                self.death_handler(),
                self.down_task,
            )
        except asyncio.exceptions.CancelledError:
            # death_handler cancelled down listener.
            pass

    async def run(self):
        subprocess_env = os.environ.copy()
        subprocess_env.update(self.proc_def.env)

        self.proc = await asyncio.create_subprocess_shell(
            util.shell_join(self.run_command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
            cwd=self.proc_def.root,
            env=subprocess_env,
            start_new_session=True,
        )
        # TODO(lshamis): Handle the case where proc dies before we can query getpgid.
        self.proc_pgrp = os.getpgid(self.proc.pid)

        life_cycle.set_state(self.name, life_cycle.State.STARTED)
        await self.gather_cmd_outputs()

    def get_pid(self):
        return self.proc.pid

    async def death_handler(self):
        await self.proc.wait()
        life_cycle.set_state(
            self.name, life_cycle.State.STOPPED, return_code=self.proc.returncode
        )
        # TODO(lshamis): Restart policy goes here.

        # Release the down listener.
        self.down_task.cancel()

    async def handle_down(self):
        try:
            if self.proc.returncode is None:
                life_cycle.set_state(self.name, life_cycle.State.STOPPING)
                self.proc.send_signal(signal.SIGTERM)

                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass

            # Clean up zombie sub-sub-processes.
            os.killpg(self.proc_pgrp, signal.SIGKILL)
        except:
            pass


class Host(BaseRuntime):
    def __init__(self, run_command, build_commands=[]):
        self.run_command = run_command
        self.build_commands = build_commands

    def asdict(self, root: pathlib.Path):
        ret = {
            "run_command": self.run_command,
        }
        if self.build_commands:
            ret["build_commands"] = self.build_commands
        return ret

    def _build(self, name: str, proc_def: ProcDef, cache: bool, verbose: bool):
        if self.build_commands:
            print(f"Building {name}")
            build_command = "\n".join(
                [util.shell_join(cmd) for cmd in self.build_commands]
            )
            result = subprocess.run(
                build_command,
                shell=True,
                executable="/bin/bash",
                capture_output=not verbose,
            )
            if result.returncode:
                raise RuntimeError(f"Failed to build: {result.stderr}")

    def _launcher(self, name: str, proc_def: ProcDef):
        return Launcher(self.run_command, name, proc_def)


__all__ = ["Host"]
