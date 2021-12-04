"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.interpreter.robot import tasks
import os
import random
import numpy as np

random.seed(2021) # fixing a random seed to fix default exploration goal
first_exploration_done = False

def get_distant_goal(x, y, t, l1_thresh=35):
    # Get a distant goal for the slam exploration
    # Pick a random quadrant, get 
    while True:
        xt = random.randint(-19, 19)
        yt = random.randint(-19, 19)
        d = np.linalg.norm(np.asarray([x,y]) - np.asarray([xt,yt]), ord=1)
        if d > l1_thresh:
            return (xt, yt, 0)

def start_explore(agent, goal):
    global first_exploration_done
    goal_str = '_'.join(str(g) for g in goal)

    if not first_exploration_done or os.getenv('CONTINUOUS_EXPLORE', 'False') == 'True':
        agent.mover.slam.reset_map()
        agent.mover.nav.reset_explore()
        task_data = { 
            "goal": goal, 
            "save_data": os.getenv('SAVE_EXPLORATION', 'False') == 'True',
            "data_path": f"{os.getenv('HEURISTIC', 'default')}/goal" + goal_str,
        }

        if os.getenv('HEURISTIC') == 'straightline':
            logging.info('Default behavior: Curious Exploration')
            agent.memory.task_stack_push(tasks.CuriousExplore(agent, task_data))
        else:
            logging.info('Default behavior: Default Exploration')
            agent.memory.task_stack_push(tasks.Explore(agent, task_data))
        
        first_exploration_done = True

def explore(agent):
    x,y,t = agent.mover.get_base_pos()
    goal = get_distant_goal(x,y,t)
    start_explore(agent, goal)