import os
import unittest

from fbrp.life_cycle import State
from fbrp.process_def import ProcDef
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
        assert c.dependencies == sorted(
            ["pytorch", "ros-noetic-genpy", "numpy", "dataclasses"]
        )
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
    async def test_envvar_for_conda(self, mock_file):
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
            env_name="fbrp_test_conda",
            run_command=["python3", "alice.py"],
            proc_def=proc_def,
        )
        os_env_patch = mock.patch.dict(os.environ, {"my_path": "path"})
        os_env_patch.start()
        conda_env = await launcher.envvar_for_conda()
        mock_file.assert_called_with(f"/tmp/fbrp_conda_test_conda.env")
        assert len(conda_env) == 1
        self.assertDictEqual(conda_env, {"env_var": "data"})
        os_env_patch.stop()

    @patch("fbrp.life_cycle.set_state")
    @patch("fbrp.runtime.conda.Launcher.gather_cmd_outputs")
    @patch("fbrp.runtime.conda.Launcher.run_cmd_with_conda_envvar")
    @patch("fbrp.runtime.conda.Launcher.envvar_for_conda")
    async def test_run(
        self,
        mock_envvar_for_conda,
        mock_run_cmd_with_conda_envvar,
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
            env_name="fbrp_test_conda",
            run_command=["python3", "alice.py"],
            proc_def=proc_def,
        )
        await launcher.run()
        mock_envvar_for_conda.assert_called_once()
        mock_run_cmd_with_conda_envvar.assert_called_once()
        mock_gather_cmd_outputs.assert_called_once()
        assert mock_set_state.call_count == 1
        mock_set_state.assert_has_calls(
            [call("test_conda", State.STARTED)]
        )


if __name__ == "__main__":
    unittest.main()
