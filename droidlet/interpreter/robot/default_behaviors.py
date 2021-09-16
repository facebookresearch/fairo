"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.interpreter.robot import tasks
import os

def explore(agent):
    if os.getenv("HEURISTIC") == 'straightline':
        logging.info("Default behavior: Curious Exploration")
        agent.memory.task_stack_push(tasks.CuriousExplore(agent, {}))
    elif os.getenv("HEURISTIC") == 'default':
        logging.info("Default behavior: Default Exploration")
        agent.memory.task_stack_push(tasks.Explore(agent, {}))
