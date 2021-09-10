import os
import logging
from agents.argument_parser import ArgumentParser
import subprocess
from multiprocessing import set_start_method
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.craftassist.make_swarm import SwarmMasterWrapper

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

def test_mc_swarm():
    from swarm_configs import get_default_config
    num_workers = 1
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Minecraft", base_path)
    opts = parser.parse()

    logging.basicConfig(level=opts.log_level.upper())

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))

    # Check that models and datasets are up to date and download latest resources.
    # Also fetches additional resources for internal users.
    if not opts.dev:
        rc = subprocess.call([opts.verify_hash_script_path, "craftassist"])

    set_start_method("spawn", force=True)
    sa = CraftAssistAgent(opts)
    master = SwarmMasterWrapper(sa, [None] * num_workers, opts, get_default_config(sa))
    master.start()

if __name__ == "__main__":
    test_mc_swarm()