import pdb, sys
from multiprocessing import Queue
from droidlet.shared_data_structs import Time
from typing import Optional, List, Tuple, Sequence, Union
from droidlet.base_util import XYZ, Block, npy_to_blocks_list
from droidlet.interpreter.task import *
from droidlet.interpreter.craftassist.tasks import *
import pickle
import uuid
from droidlet.memory.craftassist.mc_memory_nodes import (  # noqa
    # DanceNode,
    VoxelObjectNode,
    # BlockObjectNode,
    # BlockTypeNode,
    # MobNode,
    # ItemStackNode,
    # MobTypeNode,
    # InstSegNode,
    # SchematicNode,
    # NODELIST,
)

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

class SwarmWorkerMemory():
    """Represents the memory for the agent in Minecraft"""

    def __init__(self,
        memory_send_queue,
        memory_receive_queue,
        memory_tag,
    ):
        self.send_queue = memory_send_queue
        self.receive_queue = memory_receive_queue
        self.memory_tag = memory_tag
        self.receive_dict = {}
        self._safe_pickle_saved_attrs = {}
        mem_id_len = len(uuid.uuid4().hex)
        self.self_memid = "0" * (mem_id_len // 2) + uuid.uuid4().hex[: mem_id_len - mem_id_len // 2]
        
        # FIXME: insert player for locobot?
        self.db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)", self.self_memid, "Player", 0, 0, -1, False
        )
        self.tag(self.self_memid, "_physical_object")
        self.tag(self.self_memid, "_animate")
        # this is a hack until memory_filters does "not"
        self.tag(self.self_memid, "_not_location")
        self.tag(self.self_memid, "AGENT")
        # self.tag(self.self_memid, "SELF")

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

    def _db_command(self, command_name, *args):
        query_id = uuid.uuid4().hex
        send_command = [query_id, command_name]
        for a in args:
            send_command.append(a)
        self.send_queue.put(tuple(send_command))
        while query_id not in self.receive_dict.keys():
            x = self.receive_queue.get()
            self.receive_dict[x[0]] = x[1]
        to_return = self.receive_dict[query_id]
        del self.receive_dict[query_id]
        return to_return

    def get_time(self):
        return self._db_command("get_time")

    def _db_read_one(self, query:str, *args):
        return self._db_command("_db_read_one", query, *args)
    
    def _db_write(self, query: str, *args) -> int:
        if query == "INSERT INTO Tasks (uuid, action_name, pickled, prio, running, run_count, created) VALUES (?,?,?,?,?,?,?)":
            memid = args[0]
            self.tag(memid, self.memory_tag)
        to_return = self._db_command("_db_write", query, *args)
        return to_return

    def db_write(self, query: str, *args) -> int:
        return self._db_command("db_write", query, *args)
    
    def _db_read(self, query: str, *args) -> List[Tuple]:
        return self._db_command("_db_read", query, *args)
    
    def tag(self, subj_memid: str, tag_text: str):
        return self._db_command("tag", subj_memid, tag_text)

    def untag(self, subj_memid: str, tag_text: str):
        return self._db_command("untag", subj_memid, tag_text)

    def forget(self, memid):
        return self._db_command("forget", memid)

    def add_triple(self,
        subj: str = None,  # this is a memid if given
        obj: str = None,  # this is a memid if given
        subj_text: str = None,
        pred_text: str = "has_tag",
        obj_text: str = None,
        confidence: float = 1.0,
    ):
        return self._db_command("add_triple", subj, obj, subj_text, pred_text, obj_text, confidence)

    def get_triples(
        self,
        subj: str = None,
        obj: str = None,
        subj_text: str = None,
        pred_text: str = None,
        obj_text: str = None,
        return_obj_text: str = "if_exists",
    ) -> List[Tuple[str, str, str]]:
        return self._db_command("get_triples", subj, obj, subj_text, pred_text, obj_text, return_obj_text)

    def check_memid_exists(self, memid: str, table: str) -> bool:
        return self._db_command("check_memid_exists", memid, table)

    def get_mem_by_id(self, memid: str, node_type: str = None):
        return self._db_command("get_mem_by_id", memid, node_type)
    
    def basic_search(self, query):
        return self._db_command("basic_search", query)
    
    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["VoxelObjectNode"]:
        return self._db_command("get_block_object_by_xyz", xyz)
    
    def get_block_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self._db_command("get_block_object_ids_by_xyz", xyz)
    
    def get_object_info_by_xyz(self, xyz: XYZ, ref_type: str, just_memid=True):
        return self._db_command("get_object_info_by_xyz", xyz, ref_type, just_memid)
    
    def get_block_object_by_id(self, memid: str) -> "VoxelObjectNode":
        return self._db_command("get_block_object_by_id", memid)
    
    def get_object_by_id(self, memid: str, table="BlockObjects") -> "VoxelObjectNode":
        return self._db_command("get_object_by_id", memid, table)
    
    def get_instseg_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self._db_command("get_instseg_object_ids_by_xyz", xyz)
    
    def upsert_block(self,
        block: Block,
        memid: str,
        ref_type: str,
        player_placed: bool = False,
        agent_placed: bool = False,
        update: bool = True,  # if update is set to False, forces a write
    ):
        return self._db_command("upsert_block", block, memid, ref_type, player_placed, agent_placed, update)
    
    def _update_voxel_count(self, memid, dn):
        return self._db_command("_update_voxel_count", memid, dn)

    def _update_voxel_mean(self, memid, count, loc):
        return self._db_command("_update_voxel_mean", memid, count, loc)

    def remove_voxel(self, x, y, z, ref_type):
        return self._db_command("remove_voxel", self, x, y, z, ref_type)

    def set_memory_updated_time(self, memid):
        return self._db_command("set_memory_updated_time", memid)

    def set_memory_attended_time(self, memid):
        return self._db_command("set_memory_attended_time", memid)
    
    # def set_mob_position(self, mob) -> "MobNode":
    #     return self._db_command("set_mob_position", mob)
    
    def add_chat(self, speaker_memid: str, chat: str) -> str:
        return self._db_command("add_chat", speaker_memid, chat)
    
    def task_stack_peek(self) -> Optional["TaskNode"]:
        return self._db_command("task_stack_peek")
    

    



