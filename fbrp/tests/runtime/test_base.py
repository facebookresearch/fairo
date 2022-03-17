import asyncio
import unittest


from fbrp.life_cycle import Ask, ProcInfo, State
from fbrp.process_def import ProcDef
from fbrp.runtime.base import BaseLauncher

from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, Mock, patch


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


def async_return(result):
    f = asyncio.Future()
    f.set_result(result)
    return f


class TestBaseLauncher(IsolatedAsyncioTestCase):
    async def test_run(self):
        mock_proc_def = Mock(spec=ProcDef(None, None, None, None, None, None, None))
        base_launcher = BaseLauncher()
        with self.assertRaises(NotImplementedError):
            await base_launcher.run("", mock_proc_def)

    def test_get_pid(self):
        base_launcher = BaseLauncher()
        with self.assertRaises(NotImplementedError):
            base_launcher.get_pid()

    @patch("fbrp.life_cycle.aio_proc_info_watcher")
    async def test_down_watcher_1(self, mock_proc_info_watcher):
        mock_ondown = MagicMock(return_value=async_return("on down called"))
        proc_info = ProcInfo(Ask.DOWN, State.STARTED, 0, True, "")
        mock_proc_info_watcher.return_value = AsyncIter([proc_info])
        base_launcher = BaseLauncher()
        base_launcher.name = "TEST_BASE"
        await base_launcher.down_watcher(mock_ondown)
        mock_proc_info_watcher.assert_called_once()
        mock_ondown.assert_called_once()

    @patch("fbrp.life_cycle.aio_proc_info_watcher")
    async def test_down_watcher_2(self, mock_proc_info_watcher):
        mock_ondown = MagicMock(return_value=async_return("on down called"))
        proc_info = ProcInfo(Ask.UP, State.STOPPED, 0, False, "")
        mock_proc_info_watcher.return_value = AsyncIter([proc_info])
        base_launcher = BaseLauncher()
        base_launcher.name = "TEST_BASE"
        await base_launcher.down_watcher(mock_ondown)
        mock_proc_info_watcher.assert_called_once()
        mock_ondown.assert_not_called()

    @patch("fbrp.life_cycle.aio_proc_info_watcher")
    async def test_down_watcher_3(self, mock_proc_info_watcher):
        mock_ondown = MagicMock(return_value=async_return("on down called"))
        proc_info_1 = ProcInfo(Ask.UP, State.STOPPED, 0, False, "")
        proc_info_2 = ProcInfo(Ask.DOWN, State.STARTED, 0, True, "")
        mock_proc_info_watcher.return_value = AsyncIter([proc_info_1, proc_info_2])
        base_launcher = BaseLauncher()
        base_launcher.name = "TEST_BASE"
        await base_launcher.down_watcher(mock_ondown)
        mock_proc_info_watcher.assert_called_once()
        mock_ondown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
