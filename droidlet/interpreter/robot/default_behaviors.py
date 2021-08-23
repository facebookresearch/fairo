"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.interpreter.robot import tasks
import os

def explore(agent):
    logging.info("Default behavior: Exploration")
    if os.getenv("HEURISTIC") == 'straightline':
        agent.memory.task_stack_push(tasks.CuriousExplore(agent, {}))
    elif os.getenv("HEURISTIC") == 'default':
        agent.memory.task_stack_push(tasks.Explore(agent, {}))
