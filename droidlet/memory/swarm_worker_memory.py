import pdb, sys
from droidlet.memory.memory_nodes import (
    AgentNode,
    ChatNode,
    MemoryNode,
    PlayerNode,
    ProgramNode,
    ReferenceObjectNode,
    SelfNode,
    TimeNode,
    NODELIST
)
from droidlet.memory.robot.loco_memory_nodes import DetectedObjectNode
from typing import Optional, List, Tuple, Sequence, Union
from droidlet.base_util import XYZ, Block
from droidlet.task.task import *
from droidlet.interpreter.craftassist.tasks import *
import pickle
import uuid
from droidlet.memory.craftassist.mc_memory_nodes import (  # noqa
    VoxelObjectNode,
    MobNode,
    ItemStackNode,
    SchematicNode,
)

# TODO: "movement" is not picklable !!
NONPICKLE_ATTRS = [
    "agent",
    "memory",
    "agent_memory",
    "tasks_fn",
    "run_condition",
    "init_condition",
    "remove_condition",
    "stop_condition",
    "movement",
]


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class SwarmWorkerMemory:
    """Represents the memory for the agent in Minecraft"""

    def __init__(self, memory_send_queue, memory_receive_queue, memory_tag, mark_agent=False, agent_tag=None, nodelist=NODELIST, agent_time=None):
        
        self.send_queue = memory_send_queue
        self.receive_queue = memory_receive_queue
        self.memory_tag = memory_tag
        self.mark_agent = mark_agent
        self.receive_dict = {}
        self.init_time_interface(agent_time)

        mem_id_len = len(uuid.uuid4().hex)
        self._safe_pickle_saved_attrs = {}
        self.nodes = {}
        for node in nodelist:
            self.nodes[node.NODE_TYPE] = node
        # self.node_children = {}
        # for node in nodelist:
        #     self.node_children[node.NODE_TYPE] = []
        #     for possible_child in nodelist:
        #         if node in possible_child.__mro__:
        #             self.node_children[node.NODE_TYPE].append(possible_child.NODE_TYPE)
        
        # ForkedPdb().set_trace()
        self.self_memid = (
            "0" * (mem_id_len // 2) + uuid.uuid4().hex[: mem_id_len - mem_id_len // 2]
        )
        node_type = "Self"
        if self.mark_agent:
            node_type = "Agent"
        self.db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)",
            self.self_memid,
            node_type,
            0,
            0,
            -1,
            False,
        )
        # NOTE: we weren't creating this: #153-157 before.
        player_struct = None
        if node_type == "Self":
            SelfNode.create(self, player_struct, memid=self.self_memid)
        elif node_type == "Agent":
            AgentNode.create(self, player_struct, memid=self.self_memid)
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "_physical_object")
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "_animate")
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "_not_location")
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "AGENT")
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "WORKER")
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, agent_tag)

        # self.make_self_mem(agent_tag=agent_tag, agent_type="WORKER")#, self_memid=self.self_memid)

        # self.searcher = MemorySearcher()
        # if place_field_pixels_per_unit > 0:
        #     self.place_field = PlaceField(self, pixels_per_unit=place_field_pixels_per_unit)
        # else:
        #     self.place_field = EmptyPlaceField()

    def _db_command(self, command_name, *args):
        query_id = uuid.uuid4().hex
        send_command = [query_id, command_name]
        for a in args:
            send_command.append(a)
        self.send_queue.put(tuple(send_command))
        logging.info("Adding this to sed queue: %r" % (send_command))
        while query_id not in self.receive_dict.keys():
            x = self.receive_queue.get()
            self.receive_dict[x[0]] = x[1]
        to_return = self.receive_dict[query_id]
        del self.receive_dict[query_id]
        return to_return

    # def make_self_mem(self, agent_tag=None, agent_type="SELF"):#, self_memid=None):
    #     return self._db_command("make_self_mem", agent_tag, agent_type)#, self_memid)

    def init_time_interface(self, agent_time=None):
        return self._db_command("init_time_interface", agent_time)

    def get_time(self):
        return self._db_command("get_time")

    def get_world_time(self):
        return self._db_command("get_world_time")

    def add_tick(self, ticks=1):
        return self._db_command("add_tick", ticks)

    def update(self, agent):
        return self._db_command("update", agent)

    def set_memory_updated_time(self, memid):
        return self._db_command("set_memory_updated_time", memid)

    def set_memory_attended_time(self, memid):
        return self._db_command("set_memory_attended_time", memid)

    def update_recent_entities(self, mems=[]):
        return self._db_command("update_recent_entities", mems)

    def get_recent_entities(self, memtype, time_window=12000) -> List["MemoryNode"]:
        return self._db_command("get_recent_entities", memtype, time_window)

    def get_node_from_memid(self, memid: str) -> str:
        return self._db_command("get_node_from_memid", memid)

    def get_mem_by_id(self, memid: str, node_type: str = None):
        return self._db_command("get_mem_by_id", memid, node_type)

    def check_memid_exists(self, memid: str, table: str) -> bool:
        return self._db_command("check_memid_exists", memid, table)

    def forget(self, memid):
        return self._db_command("forget", memid)

    def forget_by_query(self, query: str, hard=True):
        return self._db_command("forget_by_query", query, hard)

    def basic_search(self, query):
        return self._db_command("basic_search", query)

    
    def task_stack_push(
        self, task, parent_memid: str = None, chat_effect: bool = False
    ) -> "TaskNode":
        return self._db_command("task_stack_push", task, parent_memid, chat_effect)

    def task_stack_update_task(self, memid: str, task):
        return self._db_command("task_stack_update_task", memid, task)

    def task_stack_peek(self) -> Optional["TaskNode"]:
        return self._db_command("task_stack_peek")

    def task_stack_pop(self) -> Optional["TaskNode"]:
        return self._db_command("task_stack_pop")

    def task_stack_pause(self) -> bool:
        return self._db_command("task_stack_pause")

    def task_stack_clear(self):
        return self._db_command("task_stack_clear")

    def task_stack_resume(self) -> bool:
        return self._db_command("task_stack_resume")

    def task_stack_find_lowest_instance(
        self, cls_names: Union[str, Sequence[str]]
    ) -> Optional["TaskNode"]:
        return self._db_command("task_stack_find_lowest_instance", cls_names)

    def get_last_finished_root_task(self, action_name: str = None, recency: int = None):
        return self._db_command("get_last_finished_root_task", action_name, recency)

    def _db_read(self, query: str, *args) -> List[Tuple]:
        return self._db_command("_db_read", query, *args)

    def _db_read_one(self, query: str, *args):
        return self._db_command("_db_read_one", query, *args)

    def db_write(self, query: str, *args) -> int:
        return self._db_command("db_write", query, *args)

    def _db_write(self, query: str, *args) -> int:
        if (
            query
            == "INSERT INTO Tasks (uuid, action_name, pickled, prio, running, run_count, created) VALUES (?,?,?,?,?,?,?)"
        ):
            memid = args[0]
            self.nodes[TripleNode.NODE_TYPE].tag(self, memid, self.memory_tag)
        to_return = self._db_command("_db_write", query, *args)
        return to_return

    def _db_script(self, script: str):
        return self._db_command("_db_script", script)

    def get_db_log_idx(self):
        return self._db_command("get_db_log_idx")

    def _write_to_db_log(self, s: str, *args, no_format=False):
        return self._db_command("_write_to_db_log", s, *args, no_format)

    def dump(self, sql_file, dict_memory_file=None):
        return self._db_command("dump", sql_file, dict_memory_file)

    def reinstate_attrs(self, obj):
        """
        replace non-picklable attrs on blob data, using their values
        from the key-value store, indexed by the obj memid
        """
        for attr in NONPICKLE_ATTRS:
            if hasattr(obj, "__swarm_had_attr_" + attr):
                delattr(obj, "__swarm_had_attr_" + attr)
                setattr(obj, attr, self._safe_pickle_saved_attrs[obj.memid][attr])

    def safe_unpickle(self, bs):
        """
        get non-picklable attrs from the key value store, and
        replace them on the blob data after retrieving from db
        """
        obj = pickle.loads(bs)
        self.reinstate_attrs(obj)
        return obj

    def safe_pickle(self, obj):
        """
        pickles memory objects to be put in blob data in the db.
        some attrs are not picklable, so stores these in a separate key-value store
        keyed by the memid

        """
        # little bit scary...
        for attr in NONPICKLE_ATTRS:
            if hasattr(obj, attr):
                if self._safe_pickle_saved_attrs.get(obj.memid) is None:
                    self._safe_pickle_saved_attrs[obj.memid] = {}
                val = getattr(obj, attr)
                setattr(obj, attr, None)
                setattr(obj, "__swarm_had_attr_" + attr, True)
                self._safe_pickle_saved_attrs[obj.memid][attr] = val
        p = pickle.dumps(obj)
        self.reinstate_attrs(obj)
        return p

    # ------------ minecraft agent memory commands ------------

    def update(self, perception_output=None, areas_to_perceive = []):
        return self._db_command("update", perception_output, areas_to_perceive)

    # def maybe_add_block_to_memory(self, interesting, player_placed, agent_placed, xyz, idm):
    #     return self._db_command("maybe_add_block_to_memory", interesting, player_placed, agent_placed, xyz, idm)

    # def maybe_remove_block_from_memory(self, xyz: XYZ, idm: IDM, areas_to_perceive):
    #     return self._db_command("maybe_remove_block_from_memory", xyz, idm, areas_to_perceive)

    # def maybe_remove_inst_seg(self, xyz: XYZ):
    #     return self._db_command("maybe_remove_inst_seg", xyz)

    # def add_holes_to_mem(self, holes):
    #     return self._db_command("add_holes_to_mem", holes)
    
    def get_entity_by_eid(self, eid) -> Optional["ReferenceObjectNode"]:
        return self._db_command("get_entity_by_eid", eid)

    def get_object_info_by_xyz(self, xyz: XYZ, ref_type: str, just_memid=True):
        return self._db_command("get_object_info_by_xyz", xyz, ref_type, just_memid)

    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["VoxelObjectNode"]:
        return self._db_command("get_block_object_by_xyz", xyz)

    def get_instseg_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self._db_command("get_instseg_object_ids_by_xyz", xyz)

    # def _load_mob_types(self, mobs, mob_property_data, load_mob_types=True):
    #     return self._db_command("_load_mob_types", mobs, mob_property_data, load_mob_types)

    def update_item_stack_eid(self, memid, eid) -> "ItemStackNode":
        return self._db_command("update_item_stack_eid", memid, eid)

    def set_item_stack_position(self, item_stack) -> "ItemStackNode":
        return self._db_command("set_item_stack_position", item_stack)

    def get_all_item_stacks(self):
        return self._db_command("get_all_item_stacks")

    # ------------ locobot agent memory commands ------------

    # def update_other_players(self, player_list: List):
    #     return self._db_command("update_other_players", player_list)

    # def get_detected_objects_tagged(self, *tags) -> List["DetectedObjectNode"]:
    #     return self._db_command("get_detected_objects_tagged", *tags)

    # def add_dance(self, dance_fn, name=None, tags=[]):
    #     return self._db_command("add_dance", dance_fn, name, tags)
