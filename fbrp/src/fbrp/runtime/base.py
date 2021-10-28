import asyncio
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

    def get_down_event(self):
        if not hasattr(self, "_down_requested_event"):
            self._down_requested_event = asyncio.Event()
        return self._down_requested_event

    async def command_handler(self):
        async for pkt in a0.aio_sub(
            f"fbrp/control/{self.name}", a0.INIT_AWAIT_NEW, a0.ITER_NEXT
        ):
            try:
                cmd = json.loads(pkt.payload)
                kwargs = cmd.get("kwargs", {})
                if cmd["action"] == "down":
                    self.get_down_event().set()
                    await self.handle_down(**kwargs)
            except:
                pass

    async def handle_down(self):
        raise NotImplementedError("Launcher hasn't implemented handle_down!")

    async def log_psutil(self):
        out = a0.Publisher(f"fbrp/psutil/{self.name}")
        while True:
            with contextlib.suppress(asyncio.TimeoutError):
                # TODO(lshamis): Make polling interval configurable.
                await asyncio.wait_for(self.get_down_event().wait(), 1.0)
            if self.get_down_event().is_set():
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
