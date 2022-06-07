"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
from droidlet.interpreter.robot import tasks
from droidlet.memory.memory_nodes import TripleNode
from droidlet.memory.robot.loco_memory_nodes import DetectedObjectNode
import os
import random
import numpy as np
import shutil
import json

random.seed(2021)  # fixing a random seed to fix default exploration goal


def get_distant_goal(x, y, t, l1_thresh=35):
    # Get a distant goal for the slam exploration
    # Pick a random quadrant, get
    while True:
        xt = random.randint(-19, 19)
        yt = random.randint(-19, 19)
        d = np.linalg.norm(np.asarray([x, y]) - np.asarray([xt, yt]), ord=1)
        if d > l1_thresh:
            return (xt, yt, 0)


def init_logger():
    logger = logging.getLogger("default_behavior")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("default_behavior.log", "w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s - %(funcName)s(): %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


init_logger()

# TODO: Move these utils to a suitable place - as a class method in TripleNode
def add_or_replace(agent, pred_text, obj_text):
    memids, _ = agent.memory.basic_search(f"SELECT uuid FROM Triple WHERE pred_text={pred_text}")
    assert len(memids) <= 1, f"more than 1 {len(memids)} returned"
    if len(memids) > 0:
        agent.memory.forget(memids[0])
    agent.memory.nodes[TripleNode.NODE_TYPE].create(
        agent.memory, subj=agent.memory.self_memid, pred_text=pred_text, obj_text=obj_text
    )


def get_unique_val_from_memory(agent, pred_text, typ):
    def get_default(typ):
        if typ == int:
            return 0
        if typ == str:
            return ""

    t = agent.memory.nodes[TripleNode.NODE_TYPE].get_triples(
        agent.memory, subj=agent.memory.self_memid, pred_text=pred_text
    )
    # get_triples returns a list of tuples of the form (subject, predicate, object)
    assert len(t) <= 1, f"More than 1 ({len(t)}) triple for {pred_text}"
    return typ(t[0][2]) if len(t) == 1 else get_default(typ)


def start_explore(agent, goal):
    first_exploration_done = (
        get_unique_val_from_memory(agent, "first_exploration_done", str) == "True"
    )
    explore_count = get_unique_val_from_memory(agent, "explore_count", int)

    if not first_exploration_done or os.getenv("CONTINUOUS_EXPLORE", "False") == "True":
        agent.mover.slam.reset_map()
        agent.mover.nav.reset_explore()

        logger = logging.getLogger("default_behavior")
        logger.info(
            f"Starting exploration {explore_count} \
            first_exploration_done {first_exploration_done} \
            os.getenv('CONTINUOUS_EXPLORE') {os.getenv('CONTINUOUS_EXPLORE', 'False')}"
        )

        # FIXME, don't clear the memory, place_field, etc.  explore more reasonably
        # Clear memory
        objects = DetectedObjectNode.get_all(agent.memory)
        logger.info(f"Clearing {len(objects)} memids in memory")
        agent.memory.clear(objects)
        agent.memory.place_field.clear_examined()
        # reset object id counter
        agent.perception_modules["vision"].vision.deduplicate.object_id_counter = 1
        objects = DetectedObjectNode.get_all(agent.memory)
        logger.info(f"{len(objects)} memids in memory")

        task_data = {
            "goal": goal,
            "save_data": os.getenv("SAVE_EXPLORATION", "False") == "True",
            "data_path": os.path.join(f"{os.getenv('HEURISTIC', 'default')}", str(explore_count)),
        }
        logger.info(f"task_data {task_data}")

        if os.path.isdir(task_data["data_path"]):
            shutil.rmtree(task_data["data_path"])

        if os.getenv("HEURISTIC") in ("straightline", "circle"):
            logging.info("Default behavior: Curious Exploration")
            agent.memory.task_stack_push(tasks.CuriousExplore(agent, task_data))
        else:
            logging.info("Default behavior: Default Exploration")
            agent.memory.task_stack_push(tasks.Explore(agent, task_data))

        add_or_replace(agent, "first_exploration_done", "True")
        add_or_replace(agent, "explore_count", str(explore_count + 1))


def explore(agent):
    x, y, t = agent.mover.get_base_pos()
    goal = get_distant_goal(x, y, t)
    start_explore(agent, goal)


def get_task_data(agent):
    try:
        with open(agent.opts.reexplore_json, "r") as f:
            reex = json.load(f)
    except Exception as ex:
        logging.info(f"Exception while loading {agent.opts.reexplore_json}: {ex}")
        reex = None

    reexplore_id = get_unique_val_from_memory(agent, "reexplore_id", int)
    reex_key = str(reexplore_id)
    if reex_key not in reex.keys():
        return None

    task_data = {
        "spawn_pos": reex[reex_key]["spawn_pos"],
        "base_pos": reex[reex_key]["base_pos"],
        "target": {"xyz": reex[reex_key]["target"], "label": "object"},
        "data_path": f"{agent.opts.data_store_path}/{reexplore_id}",
        "vis_path": f"{agent.opts.data_store_path}/{reexplore_id}",
    }
    add_or_replace(agent, "reexplore_id", str(reexplore_id + 1))
    return task_data


def reexplore(agent):
    task_data = get_task_data(agent)
    logging.info(f"task_data {task_data}")
    if task_data is not None:
        agent.memory.task_stack_push(tasks.Reexplore(agent, task_data))
