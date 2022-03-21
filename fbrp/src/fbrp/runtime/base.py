from fbrp import life_cycle
from fbrp.process_def import ProcDef
import a0
import asyncio
import contextlib
import json
import pathlib
import psutil


# psutil as_dict() does not produce a json-serializable dict.
# https://github.com/giampaolo/psutil/issues/967
#
# When converted to a dict, fields like memory_info generate
#     [11984896, 31031296, ...]
# instead of
#     {"rss": 11984896, "vms": 31031296, ...}
# Losing field names.
#
# We cannot use a custom JSONEncoder, since psutil objects, like memory_info,
# inherit from tuples.
#
# This is a workaround.
def _walk_asdict(obj):
    if hasattr(obj, "_asdict"):
        return _walk_asdict(obj._asdict())
    if isinstance(obj, dict):
        return {k: _walk_asdict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_asdict(v) for v in obj]
    return obj


class BaseLauncher:
    def __init__(self):
        pass

    async def run(self, name: str, proc_def: ProcDef):
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
                pkt = a0.Packet(
                    [("content-type", "application/json")],
                    json.dumps(_walk_asdict(proc.as_dict())))
                out.pub(pkt)
            except psutil.NoSuchProcess:
                pass


class BaseRuntime:
    def __init__(self):
        pass

    def asdict(self, root: pathlib.Path):
        raise NotImplementedError("Runtime hasn't implemented asdict!")

    def _build(self, name: str, proc_def: ProcDef, cache: bool, verbose: bool):
        raise NotImplementedError("Runtime hasn't implemented build!")

    def _launcher(self, name: str, proc_def: ProcDef) -> BaseLauncher:
        raise NotImplementedError("Runtime hasn't implemented a launcher!")
