import asyncio
from fbrp import life_cycle
from fbrp.process import ProcDef
import a0
import argparse
import asyncio
import contextlib
import psutil
import json


class BaseLauncher:
    def __init__(self):
        pass

    async def run(self, name: str, proc_def: ProcDef, args: argparse.Namespace):
        raise NotImplementedError("Launcher hasn't implemented run!")

    def get_pid(self):
        raise NotImplementedError("Launcher hasn't implemented get_pid!")

    async def down_watcher(self, ondown):
        async for proc_info in life_cycle.aio_proc_info_watcher(self.name):
            if proc_info.ask == life_cycle.Ask.DOWN:
                await ondown()
                break

    async def log_psutil(self):
        down_requested_event = asyncio.Event()

        async def ondown():
            down_requested_event.set()

        asyncio.ensure_future(self.down_watcher(ondown))

        out = a0.Publisher(f"fbrp/psutil/{self.name}")
        while True:
            with contextlib.suppress(asyncio.TimeoutError):
                # TODO(lshamis): Make polling interval configurable.
                await asyncio.wait_for(down_requested_event.wait(), 1.0)
            if down_requested_event.is_set():
                break
            pid = self.get_pid()
            if not pid:
                # Note: Likely being restarted.
                continue

            try:
                proc = psutil.Process(pid)
                out.pub(json.dumps(proc.as_dict()))
            except psutil.NoSuchProcess:
                pass


class BaseRuntime:
    def __init__(self):
        pass

    def _build(self, name: str, proc_def: ProcDef, args: argparse.Namespace):
        raise NotImplementedError("Runtime hasn't implemented build!")

    def _launcher(
        self, name: str, proc_def: ProcDef, args: argparse.Namespace
    ) -> BaseLauncher:
        raise NotImplementedError("Runtime hasn't implemented a launcher!")
