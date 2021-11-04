from fbrp import util
from fbrp import registrar
import a0
import argparse
import json


@registrar.register_command("ps")
class ps_cmd:
    @classmethod
    def define_argparse(cls, parser: argparse.ArgumentParser):
        pass

    @staticmethod
    def exec(args: argparse.Namespace):
        try:
            info_json = a0.Cfg("fbrp/state").read().payload
        except:
            print("No processes found.")
            return

        procs_info = json.loads(info_json)
        if all(info["state"] == "STOPPED" for info in procs_info.values()):
            print("No processes found.")
            return

        suffix_map = {
            "STARTED": "",
            "STARTING": " (starting)",
            "STOPPING": " (stopping)",
        }

        for proc, info in sorted(procs_info.items()):
            if info["state"] in suffix_map:
                print(f"{info['timestamp']}  {proc}{suffix_map[info['state']]}")
