from facebook_robotics_platform.setup.process import ProcDef
import a0
import argparse
import json


class BaseLauncher:
    def __init__(self):
        pass

    async def run(self, name: str, proc_def: ProcDef, args: argparse.Namespace):
        raise NotImplementedError("Launcher hasn't implemented run!")

    async def command_handler(self):
        async for pkt in a0.aio_sub(
            f"_/control/{self.name}", a0.INIT_AWAIT_NEW, a0.ITER_NEXT
        ):
            try:
                cmd = json.loads(pkt.payload)
                handle = {
                    "down": self.handle_down,
                }[cmd["action"]]
                await handle(**cmd.get("kwargs", {}))
            except:
                pass

    async def handle_down(self):
        raise NotImplementedError("Launcher hasn't implemented handle_down!")


class BaseRuntime:
    def __init__(self):
        pass

    def _build(self, name: str, proc_def: ProcDef, args: argparse.Namespace):
        raise NotImplementedError("Runtime hasn't implemented build!")

    def _launcher(
        self, name: str, proc_def: ProcDef, args: argparse.Namespace
    ) -> BaseLauncher:
        raise NotImplementedError("Runtime hasn't implemented a launcher!")
