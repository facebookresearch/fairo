from droidlet.interpreter.craftassist import SwarmMCInterpreter
from droidlet.interpreter.robot import SwarmLocoInterpreter

def get_default_task_info(task_map):
    DEFAULT_TASK_INFO = {
        "move": ["target"],
        "build": ["blocks_list"],
        "destroy": ["schematic"],
        "dig": ["origin", "length", "width", "depth"],
    }
    task_info = dict()
    for key in task_map:
        if key in DEFAULT_TASK_INFO.keys():
            task_info[key] = DEFAULT_TASK_INFO[key]
        else:
            task_info[key] = []
    return task_info

def get_swarm_interpreter(swarm_master_agent):
    agent_type = swarm_master_agent.__class__.__name__.lower()
    if "craft" in agent_type:
        return SwarmMCInterpreter
    elif "loco" in agent_type:
        # TODO: check if implementation works
        return SwarmLocoInterpreter
    else:
        raise NotImplementedError

def get_memory_handlers_dict(swarm_master_agent):
    agent_type = swarm_master_agent.__class__.__name__.lower()
    handle_query_dict = {
            "_db_read": swarm_master_agent.memory._db_read,
            "_db_read_one": swarm_master_agent.memory._db_read_one,
            "_db_write": swarm_master_agent.memory._db_write,
            "db_write": swarm_master_agent.memory.db_write,
            "tag": swarm_master_agent.memory.tag,
            "untag": swarm_master_agent.memory.untag,
            "forget": swarm_master_agent.memory.forget,
            "add_triple": swarm_master_agent.memory.add_triple,
            "get_triples": swarm_master_agent.memory.get_triples,
            "check_memid_exists": swarm_master_agent.memory.check_memid_exists,
            "get_mem_by_id": swarm_master_agent.memory.get_mem_by_id,
            "basic_search": swarm_master_agent.memory.basic_search,
            "get_block_object_by_xyz": swarm_master_agent.memory.get_block_object_by_xyz,
            "get_block_object_ids_by_xyz": swarm_master_agent.memory.get_block_object_ids_by_xyz,
            "get_object_info_by_xyz": swarm_master_agent.memory.get_object_info_by_xyz,
            "get_block_object_by_id": swarm_master_agent.memory.get_block_object_by_id,
            "get_object_by_id": swarm_master_agent.memory.get_object_by_id,
            "get_instseg_object_ids_by_xyz": swarm_master_agent.memory.get_instseg_object_ids_by_xyz,
            "upsert_block": swarm_master_agent.memory.upsert_block,
            "_update_voxel_count": swarm_master_agent.memory._update_voxel_count,
            "_update_voxel_mean": swarm_master_agent.memory._update_voxel_mean,
            "remove_voxel": swarm_master_agent.memory.remove_voxel,
            "set_memory_updated_time": swarm_master_agent.memory.set_memory_updated_time,
            "set_memory_attended_time": swarm_master_agent.memory.set_memory_attended_time,
            "add_chat": swarm_master_agent.memory.add_chat,
            "task_stack_peek": swarm_master_agent.memory.task_stack_peek
        }
    if "craft" in agent_type:
        handle_query_dict["get_time"] = swarm_master_agent.memory.get_time
    return handle_query_dict


def get_default_config(swarm_master_agent):
    agent_type = swarm_master_agent.__class__.__name__.lower()
    config = {
        "disable_perception_modules": []
    }
    return config



