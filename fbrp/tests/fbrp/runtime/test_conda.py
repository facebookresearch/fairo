import os
import unittest

from fbrp.life_cycle import State
from fbrp.process import ProcDef
from fbrp.runtime.conda import CondaEnv, Launcher

from unittest import IsolatedAsyncioTestCase
from unittest import mock
from unittest.mock import call, patch, mock_open


class TestCondaEnv(unittest.TestCase):
    def test_merge_and_fix_pip_1(self):
        a = CondaEnv(
            channels=["conda-forge", "robostack", "pytorch"],
            dependencies=["ros-noetic-genpy"],
        )
        b = CondaEnv(
            channels=["pytorch", "nvidia"],
            dependencies=["pytorch", "numpy", "dataclasses"],
        )
        c = CondaEnv.merge(a, b)
        assert len(c.channels) == 4
        assert c.channels == sorted(["conda-forge", "robostack", "pytorch", "nvidia"])
        assert len(c.dependencies) == 4
        assert c.dependencies == sorted(["pytorch", "ros-noetic-genpy", "numpy", "dataclasses"])
        assert "pip" not in c.dependencies
        c.fix_pip()
        assert "pip" not in c.dependencies

    def test_merge_and_fix_pip_2(self):
        a = CondaEnv(
            channels=["conda-forge", "robostack", "pytorch"],
            dependencies=["ros-noetic-genpy", {"pip": ["scipy", "cuda110", "pytorch"]}],
        )
        b = CondaEnv(
            channels=["pytorch", "nvidia"],
            dependencies=["pytorch", "numpy", "dataclasses"],
        )
        c = CondaEnv.merge(a, b)
        assert len(c.channels) == 4
        assert c.channels == sorted(["conda-forge", "robostack", "pytorch", "nvidia"])
        assert len(c.dependencies) == 5
        ref_deps = sorted(["pytorch", "ros-noetic-genpy", "numpy", "dataclasses"])
        list_deps = []
        for dep in c.dependencies:
            if type(dep) == dict:
                assert sorted(dep["pip"]) == sorted(["scipy", "cuda110", "pytorch"])
            else:
                assert dep in ref_deps
                list_deps.append(dep)
        assert "pip" not in c.dependencies
        c.fix_pip()
        assert "pip" in c.dependencies


class TestLauncher(IsolatedAsyncioTestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="env_var=data" + "\0")
    @patch("argparse.Namespace")
    async def test_activate_conda_env(self, mock_namespace, mock_file):
        proc_def = ProcDef(
            name="test_conda",
            root=None,
            rule_file=None,
            runtime="BaseRuntime",
            cfg={},
            deps=[],
            env={},
        )
        launcher = Launcher(
            name="test_conda",
            run_command=["python3", "alice.py"],
            proc_def=proc_def,
            args=mock_namespace,
        )
        os_env_patch = mock.patch.dict(os.environ, {"my_path": "path"})
        os_env_patch.start()
        conda_env = await launcher.activate_conda_env()
        mock_file.assert_called_with(f"/tmp/fbrp_conda_test_conda.env")
        assert len(conda_env) == 1
        self.assertDictEqual(conda_env, {"env_var": "data"})
        os_env_patch.stop()

    @patch("fbrp.life_cycle.set_state")
    @patch("fbrp.runtime.conda.Launcher.gather_cmd_outputs")
    @patch("fbrp.runtime.conda.Launcher.run_cmd_in_env")
    @patch("fbrp.runtime.conda.Launcher.activate_conda_env")
    @patch("argparse.Namespace")
    async def test_run(
        self,
        mock_namespace,
        mock_activate_conda_env,
        mock_run_cmd_in_env,
        mock_gather_cmd_outputs,
        mock_set_state,
    ):
        proc_def = ProcDef(
            name="test_conda",
            root=None,
            rule_file=None,
            runtime="BaseRuntime",
            cfg={},
            deps=[],
            env={},
        )
        launcher = Launcher(
            name="test_conda",
            run_command=["python3", "alice.py"],
            proc_def=proc_def,
            args=mock_namespace,
        )
        await launcher.run()
        mock_activate_conda_env.assert_called_once()
        mock_run_cmd_in_env.assert_called_once()
        mock_gather_cmd_outputs.assert_called_once()
        assert mock_set_state.call_count == 2
        mock_set_state.assert_has_calls(
            [call("test_conda", State.STARTING), call("test_conda", State.STARTED)]
        )

    @patch("fbrp.life_cycle.set_state")
    @patch("fbrp.runtime.conda.Launcher.exit_cmd_in_env")
    @patch("argparse.Namespace")
    async def test_death_handler(
        self, mock_namespace, mock_exit_cmd_in_env, mock_set_state
    ):
        mock_exit_cmd_in_env.return_value = 0
        proc_def = ProcDef(
            name="test_conda",
            root=None,
            rule_file=None,
            runtime="BaseRuntime",
            cfg={},
            deps=[],
            env={},
        )
        launcher = Launcher(
            name="test_conda",
            run_command=["python3", "alice.py"],
            proc_def=proc_def,
            args=mock_namespace,
        )
        await launcher.death_handler()
        mock_exit_cmd_in_env.assert_called_once()
        mock_set_state.assert_called_once_with(
            "test_conda", State.STOPPED, return_code=0
        )


if __name__ == "__main__":
    unittest.main()
