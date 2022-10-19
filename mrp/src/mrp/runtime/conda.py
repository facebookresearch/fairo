from mrp import life_cycle
from mrp import util
from mrp.process_def import ProcDef
from mrp.runtime.base import BaseLauncher, BaseRuntime
import asyncio
import dataclasses
import filecmp
import json
import os
import pathlib
import pty
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
    def load(flo) -> "CondaEnv":
        """Load a conda env definition from a yaml flo."""
        result = pyyaml.safe_load(flo)
        return CondaEnv(result.get("channels", []), result.get("dependencies", []))

    @staticmethod
    def from_named_env(name) -> "CondaEnv":
        """Load a conda env definition from a named environment."""
        return CondaEnv.load(
            subprocess.check_output(["conda", "env", "export", "-n", name])
        )

    @staticmethod
    def merge(lhs: "CondaEnv", rhs: "CondaEnv") -> "CondaEnv":
        """Merge two conda environments."""
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

    def fix_empty(self):
        """Fix empty dependencies.

        Conda will not create an environment if the dependencies list is empty.
        """
        if not self.dependencies:
            self.dependencies = ["python"]

    def fix_pip(self):
        """Fix pip dependencies.

        Conda special cases pip dependencies with the form:
        {"pip": ["pkg_a", "pkg_b"]}

        But still requires a top-level pip dependency.
        This method adds a top-level pip dependency if not present.
        """
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

    async def envvar_for_conda(self) -> dict:
        """Detect the envvar set by conda for our env."""
        # We grab the conda env variables separate from executing the run
        # command to simplify detecting pid and removing some race conditions.
        subprocess_env = os.environ.copy()
        subprocess_env.update(self.proc_def.env)
        envvar_file = f"/tmp/mrp_conda_{self.name}.env"
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

    async def run_cmd_with_conda_envvar(self, conda_envvar):
        """Run the command with the conda envvar."""
        self.pty_in = pty.openpty()
        self.pty_out = pty.openpty()
        self.pty_err = pty.openpty()

        self.proc = await asyncio.create_subprocess_shell(
            util.shell_join(self.run_command),
            stdin=self.pty_in[1],
            stdout=self.pty_out[1],
            stderr=self.pty_err[1],
            cwd=self.proc_def.root,
            env=conda_envvar,
            start_new_session=True,
        )
        try:
            self.proc_pgrp = os.getpgid(self.proc.pid)
            return True
        except Exception:
            return False

    async def gather_cmd_outputs(self):
        """Track the command.

        This includes:
        - piping stdout and stderr to the alephzero logger.
        - logging the processes psutil stats.
        - watching for process death.
        - watching for user down request.
        """
        self.down_task = asyncio.create_task(self.down_watcher(self.handle_down))

        log_hdl = util.LogPtyPipes(self.pty_in[0], self.pty_out[0], self.pty_err[0])
        log_hdl.start()

        try:
            await asyncio.gather(
                self.log_psutil(),
                self.death_handler(),
                self.down_task,
            )
        except asyncio.exceptions.CancelledError:
            # death_handler cancelled down listener.
            pass

        await log_hdl.stop()

    async def run(self):
        """Run the command."""
        conda_envvar = await self.envvar_for_conda()
        if await self.run_cmd_with_conda_envvar(conda_envvar):
            life_cycle.set_state(self.name, life_cycle.State.STARTED)
            await self.gather_cmd_outputs()
        life_cycle.set_state(
            self.name, life_cycle.State.STOPPED, return_code=self.proc.returncode
        )

    def get_pid(self):
        return self.proc.pid

    async def death_handler(self):
        """Watch for process death."""
        await self.proc.wait()
        # TODO(lshamis): Restart policy goes here.

        # Release the down listener.
        self.down_task.cancel()

    async def handle_down(self):
        """Handle user down request."""
        try:
            # If still running...
            if self.proc.returncode is None:
                life_cycle.set_state(self.name, life_cycle.State.STOPPING)
                # Send a gentle SIGTERM to let the process know it's time to die.
                self.proc.send_signal(signal.SIGTERM)

                try:
                    # Wait for the process to die.
                    await asyncio.wait_for(self.proc.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass

            # Send a SIGKILL to force the process group to kill stray processes.
            os.killpg(self.proc_pgrp, signal.SIGKILL)
        except:
            pass


class Conda(BaseRuntime):
    class SharedEnv:
        def __init__(
            self,
            name,
            use_named_env=None,
            copy_named_env=None,
            yaml=None,
            channels=[],
            dependencies=[],
            use_mamba=None,
            setup_commands=[],
        ):
            if use_named_env and any([copy_named_env, yaml, channels, dependencies]):
                raise ValueError(
                    "'use_named_env' cannot be mixed with other Conda environment commands."
                )

            self.name = name
            self.use_named_env = use_named_env
            self.copy_named_env = copy_named_env
            self.yaml = yaml
            self.conda_env = CondaEnv(channels[:], dependencies[:])
            self.use_mamba = (
                use_mamba if use_mamba is not None else bool(shutil.which("mamba"))
            )
            self.setup_commands = setup_commands
            self._built = False

            if use_named_env:
                self._validate_use_named_env()

        def _validate_use_named_env(self):
            """Validate that the named env exists."""
            result = subprocess.run(
                f"""
                    eval "$(conda shell.bash hook)"
                    conda activate {self.use_named_env}
                """,
                shell=True,
                executable="/bin/bash",
                stderr=subprocess.PIPE,
            )
            if result.returncode:
                raise RuntimeError(f"'use_named_env' not valid: {result.stderr}")

        def _generate_env_content(self, root: pathlib.Path) -> dict:
            """Generate the conda environment content.

            This includes the channels and dependencies obtained from:
            - explicit channels and dependencies.
            - querying a named env to duplicate.
            - given yaml file.
            """
            if self.copy_named_env:
                self.conda_env = CondaEnv.merge(
                    self.conda_env, CondaEnv.from_named_env(self.copy_named_env)
                )

            if self.yaml:
                yaml_path = os.path.join(root, self.yaml)
                self.conda_env = CondaEnv.merge(
                    self.conda_env, CondaEnv.load(open(yaml_path, "r"))
                )

            self.conda_env.fix_empty()
            self.conda_env.fix_pip()
            return {
                "channels": self.conda_env.channels,
                "dependencies": self.conda_env.dependencies,
            }

        def _env_name(self):
            """The target environment name."""
            return self.use_named_env or f"mrp_{self.name}"

        def _config_path(self):
            dirpath = os.path.expanduser(f"~/.config/mrp/conda/{self._env_name()}/")
            os.makedirs(dirpath, exist_ok=True)
            return dirpath

        def _yaml_path(self):
            return os.path.join(self._config_path(), "conda_env.yaml")

        def _setup_commands_snapshot_path(self):
            return os.path.join(self._config_path(), "setup_commands.snapshot")

        def _conda_history_snapshot_path(self):
            return os.path.join(self._config_path(), "history.snapshot")

        def _conda_history_path(self):
            info = json.loads(
                subprocess.check_output(
                    ["conda", "env", "export", "--json", "-n", self._env_name()]
                )
            )

            return os.path.join(info["prefix"], "conda-meta/history")

        def _cache_valid(self, env_content):
            try:
                # Check if the env description has changed.
                old_content = json.load(open(self._yaml_path()))
                old_content_compare_key = json.dumps(old_content, sort_keys=True)
                new_content_compare_key = json.dumps(env_content, sort_keys=True)
                if old_content_compare_key != new_content_compare_key:
                    print("detected change in environment definition.")
                    return False

                # Check if the setup commands have changed.
                if self.setup_commands != json.load(
                    open(self._setup_commands_snapshot_path())
                ):
                    print("detected change in setup commands.")
                    return False

                # Check if unmanaged commands have been executed.
                if not filecmp.cmp(
                    self._conda_history_path(), self._conda_history_snapshot_path()
                ):
                    print("detected change in conda environment.")
                    return False

                return True
            except Exception:
                return False

        def _create_env(self, root: pathlib.Path, cache: bool, verbose: bool):
            """Create the conda environment."""
            yaml_path = self._yaml_path()
            env_content = self._generate_env_content(root)
            env_content["name"] = self._env_name()

            if cache and self._cache_valid(env_content):
                return

            with open(yaml_path, "w") as env_fp:
                json.dump(env_content, env_fp, indent=2)

            print(
                f"creating conda env for {self._env_name()}. This will take a minute..."
            )

            update_bin = "mamba" if self.use_mamba else "conda"
            # https://github.com/conda/conda/issues/7279
            # Updating an existing environment does not remove old packages, even with --prune.
            subprocess.run(
                [update_bin, "env", "remove", "-n", self._env_name()],
                capture_output=not verbose,
            )
            result = subprocess.run(
                [update_bin, "env", "update", "--prune", "-f", yaml_path],
                capture_output=not verbose,
            )
            if result.returncode:
                raise RuntimeError(f"Failed to set up conda env: {result.stderr}")

            print(f"running setup commands for {self._env_name()}")

            setup_command = "\n".join(
                [util.shell_join(cmd) for cmd in self.setup_commands]
            )
            result = subprocess.run(
                f"""
                    eval "$(conda shell.bash hook)"
                    conda activate {self._env_name()}
                    cd {root}
                    {setup_command}
                """,
                shell=True,
                executable="/bin/bash",
                capture_output=not verbose,
            )
            if result.returncode:
                raise RuntimeError(f"Failed to set up conda env: {result.stderr}")

            # Snapshot successful build info.
            shutil.copy2(
                self._conda_history_path(), self._conda_history_snapshot_path()
            )
            json.dump(
                self.setup_commands, open(self._setup_commands_snapshot_path(), "w")
            )

        def _build(self, root: pathlib.Path, cache: bool, verbose: bool):
            if self._built:
                return
            if not self.use_named_env:
                self._create_env(root, cache, verbose)
            self._built = True

    def __init__(
        self,
        run_command,
        use_named_env=None,
        copy_named_env=None,
        shared_env=None,
        yaml=None,
        channels=[],
        dependencies=[],
        setup_commands=[],
        use_mamba=None,
    ):
        """Declare a conda runtime environment.

        Args:
            run_command: The command to run in the conda environment.
            use_named_env: The name of the conda environment to use. This is exclusive from other env definition arguments.
            copy_named_env: Create a new conda environment by duplicating a named environment.
            shared_env: A shared Conda.SharedEnv instance. This is exclusive from other env definition arguments.
            yaml: Create a new conda environment from the definition file.
            channels: Create a new conda environment using the given channels.
            dependencies: Create a new conda environment using the given dependencies.
            setup_commands: Commands to run during the build phase.
            use_mamba: Use mamba instead of conda. If not set, mamba will be autodetected.
        """
        if shared_env and any(
            [use_named_env, copy_named_env, yaml, channels, dependencies]
        ):
            raise ValueError(
                "'shared_env' cannot be mixed with other Conda environment commands."
            )

        self._env = shared_env
        self._is_personal_env = not shared_env
        if self._is_personal_env:
            self._env = Conda.SharedEnv(
                "__defer__",
                use_named_env,
                copy_named_env,
                yaml,
                channels,
                dependencies,
                use_mamba,
                setup_commands,
            )

        self.run_command = run_command

    def asdict(self, root: pathlib.Path):
        ret = {}
        if self._env.use_named_env:
            ret["use_named_env"] = self._env.use_named_env
        else:
            ret["env"] = self._env._generate_env_content(root)
        if self.run_command:
            ret["run_command"] = self.run_command
        return ret

    def _build(self, name: str, proc_def: ProcDef, cache: bool, verbose: bool):
        """Build the conda environment and execute any setup commands."""
        if self._is_personal_env:
            self._env.name = name
        self._env._build(proc_def.root, cache, verbose)

    def _launcher(self, name: str, proc_def: ProcDef):
        if self._env.name == "__defer__":
            self._env.name = name
        return Launcher(self.run_command, name, self._env._env_name(), proc_def)


__all__ = ["Conda"]
