from fbrp import life_cycle
from fbrp import util
from fbrp.cmd._common import CommonFlags
from fbrp.process import ProcDef
from fbrp.runtime.base import BaseLauncher, BaseRuntime
import asyncio
import dataclasses
import dataclasses
import json
import os
import pathlib
import shutil
import signal
import subprocess
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
        env_name: str,
        proc_def: ProcDef,
    ):
        self.run_command = run_command
        self.name = name
        self.env_name = env_name
        self.proc_def = proc_def

    async def activate_conda_envvar(self) -> dict:
        # We grab the conda env variables separate from executing the run
        # command to simplify detecting pid and removing some race conditions.
        subprocess_env = os.environ.copy()
        subprocess_env.update(self.proc_def.env)
        envvar_file = f"/tmp/fbrp_conda_{self.name}.env"
        envvar_info = await asyncio.create_subprocess_shell(
            f"""
                eval "$(conda shell.bash hook)"
                conda activate {self.env_name}
                cp -f /proc/self/environ {envvar_file}
            """,
            stdout=asyncio.subprocess.PIPE,
            executable="/bin/bash",
            cwd=self.proc_def.root,
            env=subprocess_env,
        )

        await envvar_info.wait()
        lines = open(envvar_file).read().split("\0")
        return dict(line.split("=", 1) for line in lines if "=" in line)

    async def run_cmd_with_envvar(self, conda_env):
        self.proc = await asyncio.create_subprocess_shell(
            util.shell_join(self.run_command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
            cwd=self.proc_def.root,
            env=conda_env,
            start_new_session=True,
        )
        # TODO(lshamis): Handle the case where proc dies before we can query getpgid.
        self.proc_pgrp = os.getpgid(self.proc.pid)

    async def gather_cmd_outputs(self):
        async def log_pipe(logger, pipe):
            while True:
                line = await pipe.readline()
                if line:
                    logger(line)
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
        life_cycle.set_state(self.name, life_cycle.State.STARTING)
        conda_env = await self.activate_conda_envvar()
        await self.run_cmd_with_envvar(conda_env)
        life_cycle.set_state(self.name, life_cycle.State.STARTED)
        await self.gather_cmd_outputs()

    def get_pid(self):
        return self.proc.pid

    async def exit_cmd_in_env(self) -> int:
        await self.proc.wait()
        return self.proc.returncode

    async def death_handler(self):
        ret_code = await self.exit_cmd_in_env()
        life_cycle.set_state(self.name, life_cycle.State.STOPPED, return_code=ret_code)
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


class Conda(BaseRuntime):
    def __init__(
        self,
        run_command,
        yaml=None,
        env=None,
        env_nosandbox=None,
        channels=[],
        dependencies=[],
        setup_commands=[],
        use_mamba=None,
    ):
        if env_nosandbox and any([yaml, env, channels, dependencies]):
            util.fail(
                f"'env_nosandbox' cannot be mixed with other Conda environment commands."
            )

        self.yaml = yaml
        self.env = env
        self.env_nosandbox = env_nosandbox
        self.run_command = run_command
        self.conda_env = CondaEnv(channels[:], dependencies[:])
        self.setup_commands = setup_commands
        self.use_mamba = (
            use_mamba if use_mamba is not None else bool(shutil.which("mamba"))
        )

        if env_nosandbox:
            self._validate_env_nosandbox()

    def _validate_env_nosandbox(self):
        result = subprocess.run(
            f"""
                eval "$(conda shell.bash hook)"
                conda activate {self.env_nosandbox}
            """,
            shell=True,
            executable="/bin/bash",
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            util.fail(f"'env_nosandbox' not valid: {result.stderr}")

    def asdict(self, root: pathlib.Path):
        ret = {}
        if self.env_nosandbox:
            ret["env_nosandbox"] = self.env_nosandbox
        else:
            ret["env"] = self._generate_env_content(root)
        if self.run_command:
            ret["run_command"] = self.run_command
        if self.setup_commands:
            ret["setup_commands"] = self.setup_commands
        return ret

    def _env_name(self, name):
        return self.env_nosandbox or f"fbrp_{name}"

    def _generate_env_content(self, root: pathlib.Path):
        if self.yaml:
            yaml_path = os.path.join(root, self.yaml)
            self.conda_env = CondaEnv.merge(
                self.conda_env, CondaEnv.load(open(yaml_path, "r"))
            )

        if self.env:
            self.conda_env = CondaEnv.merge(self.conda_env, CondaEnv.from_env(self.env))

        self.conda_env.fix_pip()
        return {
            "channels": self.conda_env.channels,
            "dependencies": self.conda_env.dependencies,
        }

    def _create_env(self, name: str, proc_def: ProcDef):
        env_path = f"/tmp/fbrp_conda_{name}.yml"

        env_content = self._generate_env_content(proc_def.root)
        env_content["name"] = self._env_name(name)

        with open(env_path, "w") as env_fp:
            json.dump(env_content, env_fp, indent=2)

        print(f"creating conda env for {name}. This will take a minute...")

        update_bin = "mamba" if self.use_mamba else "conda"
        # https://github.com/conda/conda/issues/7279
        # Updating an existing environment does not remove old packages, even with --prune.
        subprocess.run(
            [update_bin, "env", "remove", "-n", self._env_name(name)],
            capture_output=not CommonFlags.verbose,
        )
        result = subprocess.run(
            [update_bin, "env", "update", "--prune", "-f", env_path],
            capture_output=not CommonFlags.verbose,
        )
        if result.returncode:
            util.fail(f"Failed to set up conda env: {result.stderr}")

    def _build(self, name: str, proc_def: ProcDef):
        if not self.env_nosandbox:
            self._create_env(name, proc_def)

        if self.setup_commands:
            print(f"setting up conda env for {name}")
            setup_command = "\n".join(
                [util.shell_join(cmd) for cmd in self.setup_commands]
            )
            result = subprocess.run(
                f"""
                    eval "$(conda shell.bash hook)"
                    conda activate {self._env_name(name)}
                    cd {proc_def.root}
                    {setup_command}
                """,
                shell=True,
                executable="/bin/bash",
                capture_output=not CommonFlags.verbose,
            )
            if result.returncode:
                util.fail(f"Failed to set up conda env: {result.stderr}")

    def _launcher(self, name: str, proc_def: ProcDef):
        return Launcher(self.run_command, name, self._env_name(name), proc_def)


__all__ = ["Conda"]
