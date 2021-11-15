import dataclasses
from fbrp import life_cycle
from fbrp import util
from fbrp.runtime.base import BaseLauncher, BaseRuntime
from fbrp.process import ProcDef
import asyncio
import dataclasses
import json
import os
import shlex
import shutil
import signal
import subprocess
import argparse
import typing
import yaml as pyyaml


@dataclasses.dataclass
class CondaEnv:
    channels: typing.List[str]
    dependencies: typing.List[typing.Union[str, dict]]

    @staticmethod
    def load(flo):
        result = pyyaml.safe_load(flo)
        return CondaEnv(result.get("channels", []), result.get("dependencies", []))

    @staticmethod
    def from_env(name):
        return CondaEnv.load(
            subprocess.run(
                ["conda", "env", "export", "-n", name], check=True, capture_output=True
            ).stdout
        )

    @staticmethod
    def merge(lhs: "CondaEnv", rhs: "CondaEnv") -> "CondaEnv":
        channels = sorted(set(lhs.channels + rhs.channels))

        deps = []
        pip_deps = []
        for dep in lhs.dependencies + rhs.dependencies:
            if type(dep) == dict and dep.get("pip"):
                pip_deps.extend(dep["pip"])
            else:
                deps.append(dep)

        deps = sorted(set(deps))

        if pip_deps:
            pip_deps = sorted(set(pip_deps))
            deps.append({"pip": pip_deps})

        return CondaEnv(channels, deps)

    def fix_pip(self):
        if any(dep is str and dep.startswith("pip") for dep in self.dependencies):
            return
        for dep in self.dependencies:
            if type(dep) == dict and dep.get("pip"):
                self.dependencies.append("pip")
                return


class Launcher(BaseLauncher):
    def __init__(
        self,
        run_command: typing.List[str],
        name: str,
        proc_def: ProcDef,
        args: argparse.Namespace,
    ):
        self.run_command = run_command
        self.name = name
        self.proc_def = proc_def
        self.args = args

    async def run(self):
        life_cycle.set_state(self.name, life_cycle.State.STARTING)

        # We grab the conda env variables separate from executing the run
        # command to simplify detecting pid and removing some race conditions.

        envinfo = await asyncio.create_subprocess_shell(
            f"""
                eval "$(conda shell.bash hook)"
                conda activate fbrp_{self.name}
                cp /proc/self/environ /tmp/fbrp_conda_{self.name}.env
            """,
            stdout=asyncio.subprocess.PIPE,
            executable="/bin/bash",
            cwd=self.proc_def.root,
        )

        await envinfo.wait()
        lines = open(f"/tmp/fbrp_conda_{self.name}.env").read().split("\0")
        conda_env = dict(line.split("=", 1) for line in lines if "=" in line)

        cmd = self.run_command
        if type(self.run_command) == list:
            cmd = shlex.join(self.run_command)

        self.proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
            cwd=self.proc_def.root,
            env=conda_env,
        )
        life_cycle.set_state(self.name, life_cycle.State.STARTED)

        async def log_pipe(logger, pipe):
            while True:
                line = await pipe.readline()
                if line:
                    logger(line)
                else:
                    break

        await asyncio.gather(
            log_pipe(util.stdout_logger(), self.proc.stdout),
            log_pipe(util.stderr_logger(), self.proc.stderr),
            self.log_psutil(),
            self.death_handler(),
            self.down_watcher(self.handle_down),
        )

    def get_pid(self):
        return self.proc.pid

    async def death_handler(self):
        await self.proc.wait()
        life_cycle.set_state(
            self.name, life_cycle.State.STOPPED, return_code=self.proc.returncode
        )
        # TODO(lshamis): Restart policy goes here.

    async def handle_down(self):
        try:
            if self.proc.returncode is not None:
                return
            proc_pid = self.get_pid()

            life_cycle.set_state(self.name, life_cycle.State.STOPPING)
            os.kill(proc_pid, signal.SIGTERM)

            for _ in range(100):
                if not os.kill(proc_pid, 0):
                    break
                await asyncio.sleep(0.03)

            if os.kill(proc_pid, 0):
                os.kill(proc_pid, signal.SIGKILL)
        except:
            pass


class Conda(BaseRuntime):
    def __init__(
        self,
        run_command,
        yaml=None,
        env=None,
        channels=[],
        dependencies=[],
        setup_commands=[],
        use_mamba=None,
    ):
        self.yaml = yaml
        self.env = env
        self.run_command = run_command
        self.conda_env = CondaEnv(channels[:], dependencies[:])
        self.setup_commands = setup_commands
        self.use_mamba = (
            use_mamba if use_mamba is not None else bool(shutil.which("mamba"))
        )

    def _build(self, name: str, proc_def: ProcDef, args: argparse.Namespace):
        if self.yaml:
            yaml_path = os.path.join(proc_def.root, self.yaml)
            self.conda_env = CondaEnv.merge(
                self.conda_env, CondaEnv.load(open(yaml_path, "r"))
            )

        if self.env:
            self.conda_env = CondaEnv.merge(self.conda_env, CondaEnv.from_env(self.env))

        self.conda_env.fix_pip()

        env_name = f"fbrp_{name}"
        env_path = f"/tmp/fbrp_conda_{name}.yml"

        with open(env_path, "w") as env_fp:
            json.dump(
                {
                    "name": env_name,
                    "channels": self.conda_env.channels,
                    "dependencies": self.conda_env.dependencies,
                },
                env_fp,
                indent=2,
            )

        print(f"creating conda env for {name}. This will take a minute...")

        update_bin = "mamba" if self.use_mamba else "conda"
        result = subprocess.run(
            [update_bin, "env", "update", "-f", env_path],
            capture_output=not args.verbose,
        )
        if result.returncode:
            util.fail(f"Failed to set up conda env: {result.stderr}")

        if self.setup_commands:
            print(f"setting up conda env for {name}")
            setup_command = "\n".join([shlex.join(cmd) for cmd in self.setup_commands])
            result = subprocess.run(
                f"""
                    eval "$(conda shell.bash hook)"
                    conda activate {env_name}
                    cd {proc_def.root}
                    {setup_command}
                """,
                shell=True,
                executable="/bin/bash",
                capture_output=not args.verbose,
            )
            if result.returncode:
                util.fail(f"Failed to set up conda env: {result.stderr}")

    def _launcher(self, name: str, proc_def: ProcDef, args: argparse.Namespace):
        return Launcher(self.run_command, name, proc_def, args)


__all__ = ["Conda"]
