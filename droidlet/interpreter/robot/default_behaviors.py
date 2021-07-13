"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.interpreter.robot import tasks


def explore(agent):
    logging.info("Default behavior: Exploration")
    agent.memory.task_stack_push(tasks.Explore(agent, {}))
