import pickle
from droidlet.interpreter.craftassist import SwarmMCInterpreter


def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except:
        return False
    return True

class empty_object():
    def __init__(self) -> None:
        pass

def safe_object(input_object):
    if isinstance(input_object, tuple):
        tuple_len = len(input_object)
        to_return = []
        for i in range(tuple_len):
            to_return.append(safe_single_object(input_object[i]))
        return tuple(to_return)
    else:
        return safe_single_object(input_object)
        
def safe_single_object(input_object):
    if is_picklable(input_object):
        return input_object       
    all_attrs = dir(input_object)
    return_obj = empty_object()
    for attr in all_attrs:
        if attr.startswith("__"):
            continue
        if is_picklable(getattr(input_object, attr)):
            setattr(return_obj, attr, getattr(input_object, attr))
    return return_obj

def get_safe_single_object_attr_dict(input_object):
    return_dict = {}
    all_attrs = vars(input_object)
    for attr in all_attrs:
        if attr.startswith("__"):
            continue
        if is_picklable(getattr(input_object, attr)):
            return_dict[attr] = all_attrs[attr]
    return return_dict

def get_default_task_info(task_object_mapping):
    """for all tasks in task_object_mapping, find or map their 
    respective info that will be passed down in task_data. eg for :
    "move": ["target", "approx"],
    "build": ["blocks_list"],
    "destroy": ["schematic"],
    "dig": ["origin", "length", "width", "depth"],
    "dance": ["movement"] etc
    """
    DEFAULT_TASK_INFO = {
        "move": ["target", "approx"]
    }
    task_info = {}
    for key in task_object_mapping:
        if key in DEFAULT_TASK_INFO.keys():
            task_info[key] = DEFAULT_TASK_INFO[key]
        else:
            task_info[key] = []
    return task_info

def get_swarm_interpreter(agent):
    """
    Find the right interpreter based on agent type
    """
    agent_type = agent.__class__.__name__.lower()
    if agent_type == "craftassistagent":
        return SwarmMCInterpreter
    # elif agent_type == "locobotagent":
    #     # TODO: check if implementation works
    #     return SwarmLocoInterpreter
    else:
        raise NotImplementedError

def get_memory_handlers_dict(agent):
    agent_type = agent.__class__.__name__.lower()
    mem_query_to_method = {
            "get_time": agent.memory.get_time,
            "get_world_time": agent.memory.get_world_time,
            "add_tick": agent.memory.add_tick,
            "update": agent.memory.update,
            "set_memory_updated_time": agent.memory.set_memory_updated_time,
            "set_memory_attended_time": agent.memory.set_memory_attended_time,
            "update_recent_entities": agent.memory.update_recent_entities,
            "get_recent_entities": agent.memory.get_recent_entities,
            "get_node_from_memid": agent.memory.get_node_from_memid,
            "get_mem_by_id": agent.memory.get_mem_by_id,
            "check_memid_exists": agent.memory.check_memid_exists,
            "forget": agent.memory.forget,
            "forget_by_query": agent.memory.forget_by_query,
            "basic_search": agent.memory.basic_search,
            "add_triple": agent.memory.add_triple,
            "tag": agent.memory.tag,
            "untag": agent.memory.untag,
            "get_memids_by_tag": agent.memory.get_memids_by_tag,
            "get_tags_by_memid": agent.memory.get_tags_by_memid,
            "get_triples": agent.memory.get_triples,
            "add_chat": agent.memory.add_chat,
            "get_chat_by_id": agent.memory.get_chat_by_id,
            "get_chat_id": agent.memory.get_chat_id,
            "get_recent_chats": agent.memory.get_recent_chats,
            "get_most_recent_incoming_chat": agent.memory.get_most_recent_incoming_chat,
            "add_logical_form": agent.memory.add_logical_form,
            "get_logical_form_by_id": agent.memory.get_logical_form_by_id,
            "get_player_by_eid": agent.memory.get_player_by_eid,
            "get_player_by_name": agent.memory.get_player_by_name,
            "get_players_tagged": agent.memory.get_players_tagged,
            "get_player_by_id": agent.memory.get_player_by_id,
            "add_location": agent.memory.add_location,
            "get_location_by_id": agent.memory.get_location_by_id,
            "get_time_by_id": agent.memory.get_time_by_id,
            "task_stack_push": agent.memory.task_stack_push,
            "task_stack_update_task": agent.memory.task_stack_update_task,
            "task_stack_peek": agent.memory.task_stack_peek,
            "task_stack_pop": agent.memory.task_stack_pop,
            "task_stack_pause": agent.memory.task_stack_pause,
            "task_stack_clear": agent.memory.task_stack_clear,
            "task_stack_resume": agent.memory.task_stack_resume,
            "task_stack_find_lowest_instance": agent.memory.task_stack_find_lowest_instance,
            "get_last_finished_root_task": agent.memory.get_last_finished_root_task,
            "_db_read": agent.memory._db_read,
            "_db_read_one": agent.memory._db_read_one,
            "db_write": agent.memory.db_write,
            "_db_write": agent.memory._db_write,
            "_db_script": agent.memory._db_script,
            "get_db_log_idx": agent.memory.get_db_log_idx,
            "_write_to_db_log": agent.memory._write_to_db_log,
            "dump": agent.memory.dump,
        }
        
    if agent_type == "craftassistagent":
        mc_mem_name_to_method = {
            "get_entity_by_eid": agent.memory.get_entity_by_eid,
            "_update_voxel_count": agent.memory._update_voxel_count,
            "_update_voxel_mean": agent.memory._update_voxel_mean,
            "remove_voxel": agent.memory.remove_voxel,
            "upsert_block": agent.memory.upsert_block,
            "get_object_by_id": agent.memory.get_object_by_id,
            "get_object_info_by_xyz": agent.memory.get_object_info_by_xyz,
            "get_block_object_ids_by_xyz": agent.memory.get_block_object_ids_by_xyz,
            "get_block_object_by_xyz": agent.memory.get_block_object_by_xyz,
            "get_block_object_by_id": agent.memory.get_block_object_by_id,
            "tag_block_object_from_schematic": agent.memory.tag_block_object_from_schematic,
            "get_instseg_object_ids_by_xyz": agent.memory.get_instseg_object_ids_by_xyz,
            "get_schematic_by_name": agent.memory.get_schematic_by_name,
            "convert_block_object_to_schematic": agent.memory.convert_block_object_to_schematic,
            "set_mob_position": agent.memory.set_mob_position,
            "update_item_stack_eid": agent.memory.update_item_stack_eid,
            "set_item_stack_position": agent.memory.set_item_stack_position,
            "get_all_item_stacks": agent.memory.get_all_item_stacks
        }
        mem_query_to_method.update(mc_mem_name_to_method)

    elif agent_type == "locobotagent":
        robot_mem_name_to_method = {
            "update_other_players": agent.memory.update_other_players,
            "get_detected_objects_tagged": agent.memory.get_detected_objects_tagged,
            "add_dance": agent.memory.add_dance,
        }
        mem_query_to_method.update(robot_mem_name_to_method)

    return mem_query_to_method

