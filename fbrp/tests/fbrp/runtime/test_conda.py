import asyncio
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
        a = CondaEnv(channels=["conda-forge", "robostack", "pytorch"], dependencies=["ros-noetic-genpy"])
        b = CondaEnv(channels=["pytorch", "nvidia"], dependencies=["pytorch", "numpy", "dataclasses"])
        c = CondaEnv.merge(a, b)
        self.assertTrue(len(c.channels) == 4)
        self.assertListEqual(c.channels, sorted(["conda-forge", "robostack", "pytorch", "nvidia"]))
        self.assertTrue(len(c.dependencies) == 4)
        self.assertListEqual(c.dependencies, sorted(["pytorch", "ros-noetic-genpy", "numpy", "dataclasses"]))
        self.assertFalse("pip" in c.dependencies)
        c.fix_pip()
        self.assertFalse("pip" in c.dependencies)


    def test_merge_and_fix_pip_2(self):
        a = CondaEnv(channels=["conda-forge", "robostack", "pytorch"], 
                     dependencies=["ros-noetic-genpy", {"pip" : ["scipy", "cuda110", "pytorch"]}])
        b = CondaEnv(channels=["pytorch", "nvidia"], dependencies=["pytorch", "numpy", "dataclasses"])
        c = CondaEnv.merge(a, b)
        self.assertTrue(len(c.channels) == 4)
        self.assertListEqual(c.channels, sorted(["conda-forge", "robostack", "pytorch", "nvidia"]))
        self.assertTrue(len(c.dependencies) == 5)
        ref_deps = sorted(["pytorch", "ros-noetic-genpy", "numpy", "dataclasses"])
        list_deps = []
        for dep in c.dependencies:
            if type(dep) == dict:
                self.assertListEqual(sorted(dep["pip"]), sorted(["scipy", "cuda110", "pytorch"]))
            else:
                self.assertTrue(dep in ref_deps)
                list_deps.append(dep)
        self.assertFalse("pip" in c.dependencies)
        c.fix_pip()
        self.assertTrue("pip" in c.dependencies)


class TestLauncher(IsolatedAsyncioTestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="env_var=data" + "\0")
    @patch("argparse.Namespace")
    async def test_activate_conda_env(self, mock_namespace, mock_file):
        proc_def = ProcDef(name="test_conda", root=None, rule_file=None, runtime="BaseRuntime", cfg={}, deps=[], env={})
        launcher = Launcher(name="test_conda", run_command=["python3", "alice.py"], proc_def=proc_def, args=mock_namespace)
        os_env_patch = mock.patch.dict(os.environ, {"my_path": "path"})
        os_env_patch.start()
        conda_env = await launcher.activate_conda_env()
        mock_file.assert_called_with(f"/tmp/fbrp_conda_test_conda.env")
        self.assertTrue(len(conda_env) == 1)
        self.assertDictEqual(conda_env, {"env_var" : "data"})
        os_env_patch.stop()
        

    @patch("fbrp.life_cycle.set_state")
    @patch("fbrp.runtime.conda.Launcher.conda_gather")
    @patch("fbrp.runtime.conda.Launcher.conda_run")
    @patch("fbrp.runtime.conda.Launcher.activate_conda_env")
    @patch("argparse.Namespace")
    async def test_run(self, mock_namespace, mock_activate_conda_env, mock_conda_run, mock_conda_gather, mock_set_state):
        proc_def = ProcDef(name="test_conda", root=None, rule_file=None, runtime="BaseRuntime", cfg={}, deps=[], env={})
        launcher = Launcher(name="test_conda", run_command=["python3", "alice.py"], proc_def=proc_def, args=mock_namespace)
        conda_env = await launcher.run()
        mock_activate_conda_env.assert_called_once()
        mock_conda_run.assert_called_once()
        mock_conda_gather.assert_called_once()
        self.assertTrue(mock_set_state.call_count == 2)
        mock_set_state.assert_has_calls([call("test_conda", State.STARTING), call("test_conda", State.STARTED)])


if __name__ == '__main__':
    unittest.main()