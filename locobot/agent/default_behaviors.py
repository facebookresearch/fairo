"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import locobot.agent.tasks as tasks


def explore(agent):
    logging.info("Default behavior: Exploration")
    agent.memory.task_stack_push(tasks.FrontierExplore(agent, {}))
