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
            "get_time": swarm_master_agent.memory.get_time,
            "get_world_time": swarm_master_agent.memory.get_world_time,
            "add_tick": swarm_master_agent.memory.add_tick,
            "update": swarm_master_agent.memory.update,
            "set_memory_updated_time": swarm_master_agent.memory.set_memory_updated_time,
            "set_memory_attended_time": swarm_master_agent.memory.set_memory_attended_time,
            "update_recent_entities": swarm_master_agent.memory.update_recent_entities,
            "get_recent_entities": swarm_master_agent.memory.get_recent_entities,
            "get_node_from_memid": swarm_master_agent.memory.get_node_from_memid,
            "get_mem_by_id": swarm_master_agent.memory.get_mem_by_id,
            "check_memid_exists": swarm_master_agent.memory.check_memid_exists,
            "forget": swarm_master_agent.memory.forget,
            "forget_by_query": swarm_master_agent.memory.forget_by_query,
            "basic_search": swarm_master_agent.memory.basic_search,
            "add_triple": swarm_master_agent.memory.add_triple,
            "tag": swarm_master_agent.memory.tag,
            "untag": swarm_master_agent.memory.untag,
            "get_memids_by_tag": swarm_master_agent.memory.get_memids_by_tag,
            "get_tags_by_memid": swarm_master_agent.memory.get_tags_by_memid,
            "get_triples": swarm_master_agent.memory.get_triples,
            "add_chat": swarm_master_agent.memory.add_chat,
            "get_chat_by_id": swarm_master_agent.memory.get_chat_by_id,
            "get_chat_id": swarm_master_agent.memory.get_chat_id,
            "get_recent_chats": swarm_master_agent.memory.get_recent_chats,
            "get_most_recent_incoming_chat": swarm_master_agent.memory.get_most_recent_incoming_chat,
            "add_logical_form": swarm_master_agent.memory.add_logical_form,
            "get_logical_form_by_id": swarm_master_agent.memory.get_logical_form_by_id,
            "get_player_by_eid": swarm_master_agent.memory.get_player_by_eid,
            "get_player_by_name": swarm_master_agent.memory.get_player_by_name,
            "get_players_tagged": swarm_master_agent.memory.get_players_tagged,
            "get_player_by_id": swarm_master_agent.memory.get_player_by_id,
            "add_location": swarm_master_agent.memory.add_location,
            "get_location_by_id": swarm_master_agent.memory.get_location_by_id,
            "get_time_by_id": swarm_master_agent.memory.get_time_by_id,
            "task_stack_push": swarm_master_agent.memory.task_stack_push,
            "task_stack_update_task": swarm_master_agent.memory.task_stack_update_task,
            "task_stack_peek": swarm_master_agent.memory.task_stack_peek,
            "task_stack_pop": swarm_master_agent.memory.task_stack_pop,
            "task_stack_pause": swarm_master_agent.memory.task_stack_pause,
            "task_stack_clear": swarm_master_agent.memory.task_stack_clear,
            "task_stack_resume": swarm_master_agent.memory.task_stack_resume,
            "task_stack_find_lowest_instance": swarm_master_agent.memory.task_stack_find_lowest_instance,
            "get_last_finished_root_task": swarm_master_agent.memory.get_last_finished_root_task,
            "_db_read": swarm_master_agent.memory._db_read,
            "_db_read_one": swarm_master_agent.memory._db_read_one,
            "db_write": swarm_master_agent.memory.db_write,
            "_db_write": swarm_master_agent.memory._db_write,
            "_db_script": swarm_master_agent.memory._db_script,
            "get_db_log_idx": swarm_master_agent.memory.get_db_log_idx,
            "_write_to_db_log": swarm_master_agent.memory._write_to_db_log,
            "dump": swarm_master_agent.memory.dump,
        }
        
    if "craft" in agent_type:
        mc_handle_query_dict = {
            "get_entity_by_eid": swarm_master_agent.memory.get_entity_by_eid,
            "_update_voxel_count": swarm_master_agent.memory._update_voxel_count,
            "_update_voxel_mean": swarm_master_agent.memory._update_voxel_mean,
            "remove_voxel": swarm_master_agent.memory.remove_voxel,
            "upsert_block": swarm_master_agent.memory.upsert_block,
            "get_object_by_id": swarm_master_agent.memory.get_object_by_id,
            "get_object_info_by_xyz": swarm_master_agent.memory.get_object_info_by_xyz,
            "get_block_object_ids_by_xyz": swarm_master_agent.memory.get_block_object_ids_by_xyz,
            "get_block_object_by_xyz": swarm_master_agent.memory.get_block_object_by_xyz,
            "get_block_object_by_id": swarm_master_agent.memory.get_block_object_by_id,
            "tag_block_object_from_schematic": swarm_master_agent.memory.tag_block_object_from_schematic,
            "get_instseg_object_ids_by_xyz": swarm_master_agent.memory.get_instseg_object_ids_by_xyz,
            "get_schematic_by_name": swarm_master_agent.memory.get_schematic_by_name,
            "convert_block_object_to_schematic": swarm_master_agent.memory.convert_block_object_to_schematic,
            "set_mob_position": swarm_master_agent.memory.set_mob_position,
            "update_item_stack_eid": swarm_master_agent.memory.update_item_stack_eid,
            "set_item_stack_position": swarm_master_agent.memory.set_item_stack_position,
            "get_all_item_stacks": swarm_master_agent.memory.get_all_item_stacks
        }
        handle_query_dict.update(mc_handle_query_dict)

    elif "loco" in agent_type:
        loco_handle_query_dict = {
            "update_other_players": swarm_master_agent.memory.update_other_players,
            "get_detected_objects_tagged": swarm_master_agent.memory.get_detected_objects_tagged,
            "add_dance": swarm_master_agent.memory.add_dance,
        }
        handle_query_dict.update(loco_handle_query_dict)

    return handle_query_dict


def get_default_config(swarm_master_agent):
    agent_type = swarm_master_agent.__class__.__name__.lower()
    config = {
        "disable_perception_modules": []
    }
    return config



