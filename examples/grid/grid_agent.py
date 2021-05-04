import os
import sys
import logging
import time
import numpy as np

from base_agent.argument_parser import ArgumentParser

from base_agent.core import BaseAgent
from base_agent.sql_memory import AgentMemory
from heuristic_perception import HeuristicPerception

from world import World
from tasks import Catch

log_formatter = logging.Formatter(
    "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s"
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers.clear()

class GridAgent(BaseAgent):
    def __init__(self, world=None, opts=None):
        self.world = world
        self.last_task_memid = None
        self.pos = (0, 0, 0)
        super(GridAgent, self).__init__(opts)

    def init_memory(self):
        self.memory = AgentMemory()

    def init_perception(self):
        self.perception_modules = {}
        self.perception_modules['heuristic'] = HeuristicPerception(self)

    def init_controller(self):
        pass
    
    def perceive(self):
        self.world.step() # update world state
        for perception_module in self.perception_modules.values():
            perception_module.perceive()

    def controller_step(self):
        bot_memids = self.memory.get_memids_by_tag("bot")
        if self.memory.task_stack_peek() is None:
            if bot_memids:            
                task_data = {"target_memid": bot_memids[0]}
                self.memory.task_stack_push(Catch(self, task_data))
                logging.info(f"pushed Catch Task of bot with memid: {bot_memids[0]}")
            else:
                exit()
    
    def task_step(self, sleep_time=5):
        while (
            self.memory.task_stack_peek() and self.memory.task_stack_peek().task.check_finished()
        ):
            self.memory.task_stack_pop()

        # do nothing if there's no task
        if self.memory.task_stack_peek() is None:
            return

        # If something to do, step the topmost task
        task_mem = self.memory.task_stack_peek()
        if task_mem.memid != self.last_task_memid:
            logging.info("Starting task {}".format(task_mem.task))
            self.last_task_memid = task_mem.memid
        task_mem.task.step(self)
        self.memory.task_stack_update_task(task_mem.memid, task_mem.task)
        self.world.visualize(self)
    
    def handle_exception(self, e):
        logging.exception("Default handler caught exception")
        self.send_chat("Oops! I got confused and wasn't able to complete my last task :(")
        self.memory.task_stack_clear()
        # self.dialogue_manager.dialogue_stack.clear()
    
    def send_chat(self, chat):
        logging.info(f"[Agent]: {chat}")


    
    """physical interfaces"""
    def get_pos(self):
        return self.pos
    
    def move(self, x, y, z):
        self.pos = (x, y, z)
        print(f"[Agent]: I moved to: ({x}, {y}, {z})")
        return self.pos
    
    def catch(self, target_eid):
        bots = self.world.get_bots(eid=target_eid)
        if len(bots) > 0:
            bot = bots[0]
            if np.linalg.norm(np.subtract(self.pos, bot.pos)) < 1.0001:
                self.world.remove_bot(target_eid)
                print(f"[Agent]: Great! I caught the bot with eid [{target_eid}]")
        


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    parser = ArgumentParser("Minecraft", base_path)
    opts = parser.parse()
    NUM_BOTS = 2
    opts.num_bots = NUM_BOTS

    # set up stdout logging
    sh = logging.StreamHandler()
    sh.setFormatter(log_formatter)
    logging.getLogger().addHandler(sh)
    logging.info("Info logging")
    logging.debug("Debug logging")


    world = World(opts)
    grid = GridAgent(world=world, opts=opts)
    grid.start()
