import argparse
import logging
import os
from agents.argument_parser import ArgumentParser
from agents.craftassist.craftassist_agent import CraftAssistAgent
from agents.make_agent_swarm import SwarmMasterWrapper
from multiprocessing import set_start_method


log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)

def make_swarm_minecraft():
    # set up opts 
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Minecraft", base_path)
    opts = parser.parse()
    # num_workers = opts.num_workers
    # set up logging
    logging.basicConfig(level=opts.log_level.upper())
    

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.addHandler(sh)
    logging.info("LOG LEVEL: {}".format(logger.level))
    num_workers = opts.num_workers
    logging.info("Number of workers = %r" % (num_workers))
   
    # this is the method that will spawn other processes
    set_start_method("spawn", force=True)
    opts.name = "swarm_master_bot"
    opts.mark_agent = True
    opts.agent_tag = "beta"
    swarm_master_agent = CraftAssistAgent(opts) 
    # Init the wrapper around the master agent with above agent, opts and 
    master = SwarmMasterWrapper(agent=swarm_master_agent, worker_agents=[None] * num_workers, opts=opts)
    master.start()


if __name__ == "__main__":
    make_swarm_minecraft()
