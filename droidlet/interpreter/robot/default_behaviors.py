"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.interpreter.robot import tasks
from droidlet.memory.robot.loco_memory_nodes import DetectedObjectNode
from droidlet.lowlevel.robot_mover_utils import ExaminedMap
import os
import random
import numpy as np
import shutil

random.seed(2021) # fixing a random seed to fix default exploration goal

def get_distant_goal(x, y, t, l1_thresh=35):
    # Get a distant goal for the slam exploration
    # Pick a random quadrant, get 
    while True:
        xt = random.randint(-19, 19)
        yt = random.randint(-19, 19)
        d = np.linalg.norm(np.asarray([x,y]) - np.asarray([xt,yt]), ord=1)
        if d > l1_thresh:
            return (xt, yt, 0)

def init_logger():
    logger = logging.getLogger('default_behavior')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('default_behavior.log', 'w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)s - %(funcName)s(): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

init_logger()

def start_explore(agent, goal):
    first_exploration_done = False
    t = agent.memory.get_triples(subj=agent.memory.self_memid, pred_text="first_exploration_done")
    assert len(t) <= 1, f"More than 1 ({len(t)}) triple for first_exploration_done"
    if len(t) == 1:
        first_exploration_done = t[0][2] == 'True'  

    explore_count = 0
    t = agent.memory.get_triples(subj=agent.memory.self_memid, pred_text="explore_count")
    assert len(t) <= 1, f"More than 1 ({len(t)}, {t}) triple for explore_count"
    if len(t) == 1:
        explore_count = int(t[0][2])
   
    if not first_exploration_done or os.getenv('CONTINUOUS_EXPLORE', 'False') == 'True':
        agent.mover.slam.reset_map()
        agent.mover.nav.reset_explore()

        logger = logging.getLogger('default_behavior')
        logger.info(
            f"Starting exploration {explore_count} \
            first_exploration_done {first_exploration_done} \
            os.getenv('CONTINUOUS_EXPLORE') {os.getenv('CONTINUOUS_EXPLORE', 'False')}"
        )

        # Clear memory
        objects = DetectedObjectNode.get_all(agent.memory)
        logger.info(f'Clearing {len(objects)} memids in memory')
        agent.memory.clear(objects)
        ExaminedMap.clear()
        # reset object id counter
        agent.perception_modules["vision"].vision.deduplicate.object_id_counter = 1
        objects = DetectedObjectNode.get_all(agent.memory)
        logger.info(f'{len(objects)} memids in memory')
        
        task_data = { 
            "goal": goal, 
            "save_data": os.getenv('SAVE_EXPLORATION', 'False') == 'True',
            "data_path": os.path.join(f"{os.getenv('HEURISTIC', 'default')}", str(explore_count)),
        }
        logger.info(f'task_data {task_data}')
        
        if os.path.isdir(task_data['data_path']):
            shutil.rmtree(task_data['data_path'])

        if os.getenv('HEURISTIC') in ('straightline', 'circle'):
            logging.info('Default behavior: Curious Exploration')
            agent.memory.task_stack_push(tasks.CuriousExplore(agent, task_data))
        else:
            logging.info('Default behavior: Default Exploration')
            agent.memory.task_stack_push(tasks.Explore(agent, task_data))
        
        def add_or_replace(agent, pred_text, obj_text):
            memids, _ = agent.memory.basic_search(f'SELECT uuid FROM Triple WHERE pred_text={pred_text}')
            assert len(memids) <= 1, f"more than 1 {len(memids)} returned"
            if len(memids) > 0:
                agent.memory.forget(memids[0])
            agent.memory.add_triple(subj=agent.memory.self_memid, pred_text=pred_text, obj_text=obj_text)

        add_or_replace(agent, 'first_exploration_done', 'True')
        add_or_replace(agent, 'explore_count', str(explore_count+1))

def explore(agent):
    x,y,t = agent.mover.get_base_pos()
    goal = get_distant_goal(x,y,t)
    start_explore(agent, goal)
