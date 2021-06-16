"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

###TODO put dances back
import gzip
import logging
import numpy as np
import os
import pickle
import sqlite3
import uuid
import droidlet.event.dispatcher as dispatch
from itertools import zip_longest
from typing import cast, Optional, List, Tuple, Sequence, Union
from droidlet.base_util import XYZ
from droidlet.shared_data_structs import Time
from droidlet.memory.memory_filters import BasicMemorySearcher
from .dialogue_stack import DialogueStack
from droidlet.memory.memory_util import parse_sql, format_query

from droidlet.memory.memory_nodes import (  # noqa
    TaskNode,
    TripleNode,
    PlayerNode,
    ProgramNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    ReferenceObjectNode,
    NamedAbstractionNode,
    NODELIST,
)

# FIXME set these in the Task classes
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

SCHEMAS = [os.path.join(os.path.dirname(__file__), "base_memory_schema.sql")]

# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class AgentMemory:
    """This class represents agent's memory and can be extended to add more
    capabilities based on the agent's framework.

    Args:
        db_file (string): The database file
        schema_path (string): Path to the file containing the database schema
        db_log_path (string): Path to where the database logs will be written
        nodelist (list[MemoryNode]): List of memory nodes
        agent_time (Time object): object with a .get_time(), get_world_hour, and add_tick()
                                   methods
        on_delete_callback (callable): callable to be run when a memory is deleted from Memories table

    Attributes:
        _db_log_file (FileHandler): File handler for writing database logs
        _db_log_idx (int): Database log index
        db (object): connection object to the database file
        _safe_pickle_saved_attrs (dict): Dictionary for pickled attributes
        all_tables (list): List of all table names
        nodes (dict): Mapping of node name to table name
        self_memid (str): MemoryID for the AgentMemory
        basic_searcher (BasicMemorySearcher): A class to search through memory
        time (int): The time of the agent
    """

    def __init__(
        self,
        db_file=":memory:",
        schema_paths=SCHEMAS,
        coordinate_transforms=None,
        db_log_path=None,
        nodelist=NODELIST,
        agent_time=None,
        on_delete_callback=None,
    ):
        if db_log_path:
            self._db_log_file = gzip.open(db_log_path + ".gz", "w")
            self._db_log_idx = 0
        if os.path.isfile(db_file):
            os.remove(db_file)
        self.db = sqlite3.connect(db_file, check_same_thread=False)
        self.dialogue_stack = DialogueStack()
        self.task_db = {}
        self._safe_pickle_saved_attrs = {}

        self.on_delete_callback = on_delete_callback

        self.init_time_interface(agent_time)

        self._dispatch_signal = dispatch.Signal()

        # FIXME agent : should this be here?  where to put?
        self.coordinate_transforms = coordinate_transforms

        for schema_path in schema_paths:
            with open(schema_path, "r") as f:
                self._db_script(f.read())

        self.all_tables = [
            c[0] for c in self._db_read("SELECT name FROM sqlite_master WHERE type='table';")
        ]
        self.nodes = {}
        for node in nodelist:
            self.nodes[node.NODE_TYPE] = node

        # create a "self" memory to reference in Triples
        self.self_memid = "0" * len(uuid.uuid4().hex)
        self.db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)", self.self_memid, "Self", 0, 0, -1, False
        )
        self.tag(self.self_memid, "_physical_object")
        self.tag(self.self_memid, "_animate")
        # this is a hack until memory_filters does "not"
        self.tag(self.self_memid, "_not_location")
        self.tag(self.self_memid, "AGENT")
        self.tag(self.self_memid, "SELF")

        self.basic_searcher = BasicMemorySearcher(self_memid=self.self_memid)

    def __del__(self):
        """Close the database file"""
        if getattr(self, "_db_log_file", None):
            self._db_log_file.close()

    def init_time_interface(self, agent_time=None):
        """Initialiaze the current time in memory

        Args:
            agent_time (int): value of time from agent process
        """
        self.time = agent_time or Time()

    def get_time(self):
        """Get current time

        Returns:
            int: current time from memory
        """
        return self.time.get_time()

    def get_world_time(self):
        """Get the current time in game world

        Returns:
            int: current time in the environment world
        """
        return self.time.get_world_time()

    def add_tick(self, ticks=1):
        """Add a tick to time to increment it

        Args:
            ticks (int): number of ticks
        """
        self.time.add_tick(ticks)

    # TODO list of all "updatable" mems, do a mem.update() ?
    def update(self, agent):
        pass

    ########################
    ### Workspace memory ###
    ########################

    def set_memory_updated_time(self, memid):
        """ "Set the updated_time of the memory object with given memid

        Args:
            memid (string): Memory ID

        Returns:
            int: Number of affected rows

        Examples::
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> set_memory_updated_time(memid)
        """
        self._db_write("UPDATE Memories SET updated_time=? WHERE uuid=?", self.get_time(), memid)

    def set_memory_attended_time(self, memid):
        """ "Set the attended_time of the memory object with given memid

        Args:
            memid (string): Memory ID

        Returns:
            int: Number of affected rows

        Examples::
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> set_memory_attended_time(memid)
        """
        self._db_write("UPDATE Memories SET attended_time=? WHERE uuid=?", self.get_time(), memid)

    def update_recent_entities(self, mems=[]):
        """ "Update memories in mems as recently attended

        Args:
            mems (list): List of memories

        Examples::
            >>> mems = ['10517cc584844659907ccfa6161e9d32',
                        '3493128492859dfksdfhs34839458934']
            >>> update_recent_entities(mems)
        """
        logging.info("update_recent_entities {}".format(mems))
        for mem in mems:
            mem.update_recently_attended()

    # for now, no archives in recent entities
    def get_recent_entities(self, memtype, time_window=12000) -> List["MemoryNode"]:
        """Get all entities of given memtype that were recently (within the
        time window) attended

        Args:
            memtype (string): The node type of memory
            time_window (int): The time window for maintaining recency window from current time

        Returns:
            list[MemoryNode]: list of MemoryNode objects

        Examples ::
            >>> memtype = 'Player'
            >>> get_recent_entities(memtype)
        """
        r = self._db_read(
            """SELECT uuid
            FROM Memories
            WHERE node_type=? AND attended_time >= ? and is_snapshot=0
            ORDER BY attended_time DESC""",
            memtype,
            self.get_time() - time_window,
        )
        return [self.get_mem_by_id(memid, memtype) for memid, in r]

    ###############
    ### General ###
    ###############

    def get_node_from_memid(self, memid: str) -> str:
        """Given the memid, return the node type

        Args:
            memid (string): Memory ID

        Returns:
            string: The node type of memory node

        Examples::
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> get_node_from_memid(memid)
        """
        (r,) = self._db_read_one("SELECT node_type FROM Memories WHERE uuid=?", memid)
        return r

    def get_mem_by_id(self, memid: str, node_type: str = None) -> "MemoryNode":
        """Given the memid and an optional node_type,
        return the memory node

        Args:
            memid (string): Memory ID
            node_type (string): the type of memory node

        Returns:
            MemoryNode: a memory node object

        Examples::
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> node_type = 'Chat'
            >>> get_mem_by_id(memid, node_type)
        """
        if node_type is None:
            node_type = self.get_node_from_memid(memid)

        if node_type is None:
            return MemoryNode(self, memid)

        return self.nodes.get(node_type, MemoryNode)(self, memid)

    # FIXME! make table optional
    def check_memid_exists(self, memid: str, table: str) -> bool:
        """Given the table and memid, check if an entry exists

        Args:
            memid (string): Memory id
            table (string): Name of table

        Returns:
            bool: whther an object with the memory id exists

        Examples::
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> table = 'ReferenceObjects'
            >>> check_memid_exists(memid, table)
        """
        return bool(self._db_read_one("SELECT * FROM {} WHERE uuid=?".format(table), memid))

    # TODO forget should be a method of the memory object
    def forget(self, memid: str):
        """remove a memory from the DB. Warning: some of the work is done by
           delete cascades in SQL

        Args:
            memid (string): Memory id

        Examples::
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> forget(memid)
        """
        self.db_write("DELETE FROM Memories WHERE uuid=?", memid)
        # TRIGGERs in the db clean up triples referencing the memid.
        # TODO this less brutally.  might want to remember some
        # triples where the subject or object has been removed
        # eventually we might have second-order relations etc, this could set
        # off a chain reaction

    def forget_by_query(self, query: str, hard=True):
        """remove memories from DB that match a query

        Args:
            query (string): should be "SELECT uuid FROM ... "
            hard (bool): flag indicating whether it is a hard delete
                         A 'soft' delete is just tagging the memory as _forgotten
        """
        qsplit = query.split()
        assert qsplit[0].lower() == "select"
        assert qsplit[1].lower() == "uuid"
        uuids = self._db_read(query)
        for u in uuids:
            self.forget(u[0])

    def basic_search(self, filter_dict):
        """Perform a basic search using the filter_dict

        Args:
            filter_dict (dict): A dictionary indicating values that the memory should be filtered on

        Returns:
            list[MemoryNode]: A list of MemoryNode objects.

        Examples::
            >>> filters_dict = {"base_table" : "ReferenceObject",
                                "triples" : [{"pred_text" : "has_name",
                                              "obj_text" : "house"}]}
            >>> basic_search(filters_dict)
        """
        return self.basic_searcher.search(self, search_data=filter_dict)

    #################
    ###  Triples  ###
    #################

    def add_triple(
        self,
        subj: str = None,  # this is a memid if given
        obj: str = None,  # this is a memid if given
        subj_text: str = None,
        pred_text: str = "has_tag",
        obj_text: str = None,
        confidence: float = 1.0,
    ):
        """Adds (subj, pred, obj) triple to the triplestore.
            _text is the name field of a NamedAbstraction; if
            such a NamedAbstraction does not exist, this builds it as a side effect.
            subj and obj can be memids or text, but pred_text is required

        Args:
            subj (string): memid of subject
            obj (string): memid of object
            subj_text (string): text representation for subject
            pred_text (string): predicate text
            obj_text (string): text representation for object
            confidence (float): The confidence score for the triple

        Returns:
            memid of triple

        Examples::
            >>> subj = '10517cc584844659907ccfa6161e9d32'
            >>> obj_text = 'blue'
            >>> pred_text = "has_colour"
            >>> add_triple(subj=subj, pred_text=pred_text, obj_text=obj_text)

        """
        return TripleNode.create(
            self,
            subj=subj,
            obj=obj,
            subj_text=subj_text,
            pred_text=pred_text,
            obj_text=obj_text,
            confidence=confidence,
        )

    def tag(self, subj_memid: str, tag_text: str):
        """Tag the subject with tag text.

        Args:
            subj_memid (string): memid of subject
            tag_text (string): string representation of the tag

        Returns:
            memid of triple representing the tag

        Examples::
            >>> subj_memid = '10517cc584844659907ccfa6161e9d32'
            >>> tag_text = "shiny"
            >>> tag(subj_memid, tag_text)
        """
        return self.add_triple(subj=subj_memid, pred_text="has_tag", obj_text=tag_text)

    # TODO remove_triple
    def untag(self, subj_memid: str, tag_text: str):
        """Delete tag for subject

        Args:
            subj_memid (string): memid of subject
            tag_text (string): string representation of the tag

        Examples::
            >>> subj_memid = '10517cc584844659907ccfa6161e9d32'
            >>> tag_text = "shiny"
            >>> untag(subj_memid, tag_text)
        """
        # FIXME replace me with a basic filters when _self handled better
        triple_memids = self._db_read(
            'SELECT uuid FROM Triples WHERE subj=? AND pred_text="has_tag" AND obj_text=?',
            subj_memid,
            tag_text,
        )
        if triple_memids:
            self.forget(triple_memids[0][0])

    # does not search archived mems for now
    # assumes tag is tag text
    def get_memids_by_tag(self, tag: str) -> List[str]:
        """Find all memids with a given tag

        Args:
            tag (string): string representation of the tag

        Returns:
            list[string]: list of memory ids (which are strings)

        Examples::
            >>> tag = "round"
            >>> get_memids_by_tag(tag)
        """
        r = self._db_read(
            'SELECT DISTINCT(Memories.uuid) FROM Memories INNER JOIN Triples as T ON T.subj=Memories.uuid WHERE T.pred_text="has_tag" AND T.obj_text=? AND Memories.is_snapshot=0',
            tag,
        )
        return [x for (x,) in r]

    def get_tags_by_memid(self, subj_memid: str, return_text: bool = True) -> List[str]:
        """Find all tag for a given memid

        Args:
            subj_memid (string): the subject's memid (uuid from Memories table)
            return_text (bool): if true, return the object text, otherwise return object memid

        Returns:
            list[string]: list of tags.

        Examples::
            >>> subj_memid = '10517cc584844659907ccfa6161e9d32'
            >>> get_tags_by_memid(subj_memid=subj_memid, return_text=True)
        """
        if return_text:
            return_clause = "obj_text"
        else:
            return_clause = "obj"
        q = (
            "SELECT DISTINCT("
            + return_clause
            + ') FROM Triples WHERE pred_text="has_tag" AND subj=?'
        )
        r = self._db_read(q, subj_memid)
        return [x for (x,) in r]

    # does not search archived mems for now
    # TODO clean up input?
    def get_triples(
        self,
        subj: str = None,
        obj: str = None,
        subj_text: str = None,
        pred_text: str = None,
        obj_text: str = None,
        return_obj_text: str = "if_exists",
    ) -> List[Tuple[str, str, str]]:
        """gets triples from the triplestore.
        subj is always returned as a memid even when searched as text.
        need at least one non-None part of the triple, and
        text should not not be input for a part of a triple where a memid is set.

        Args:
            subj (string): memid of subject
            obj (string): memid of object
            subj_text (string): text of the subject (if applicable, as opposed to subject memid)
            pred_text (string): text of the predicate
            obj_text (string): text of the subject (if applicable, as opposed to subject memid)
            return_obj_text (string): if return_obj_text == "if_exists", will return the obj_text
                             if it exists, and the memid otherwise. If return_obj_text
                             == "always", returns the obj_text even if it is None. If
                             return_obj_text == "never", returns the obj memid.

        Returns:
            list[tuple]: A list of tuples of the form : (subject, predicate, object)

        Examples::
            >>> subj = '10517cc584844659907ccfa6161e9d32'
            >>> obj_text = 'blue'
            >>> pred_text = "has_colour"
            >>> get_triples(subj=subj, pred_text=pred_text, obj_text=obj_text)
        """
        assert any([subj or subj_text, pred_text, obj or obj_text])
        # search by memid or by text, but not both
        assert not (subj and subj_text)
        assert not (obj and obj_text)
        pairs = [
            ("subj", subj),
            ("subj_text", subj_text),
            ("pred_text", pred_text),
            ("obj", obj),
            ("obj_text", obj_text),
        ]
        args = [x[1] for x in pairs if x[1] is not None]
        where = [x[0] + "=?" for x in pairs if x[1] is not None]
        if len(where) == 1:
            where_clause = where[0]
        else:
            where_clause = " AND ".join(where)
        return_clause = "subj, pred_text, obj, obj_text "
        sql = (
            "SELECT "
            + return_clause
            + "FROM Triples INNER JOIN Memories as M ON Triples.subj=M.uuid WHERE M.is_snapshot=0 AND "
            + where_clause
        )
        r = self._db_read(sql, *args)
        # subj is always returned as memid, even if pred and obj are returned as text
        # pred is always returned as text
        if return_obj_text == "if_exists":
            l = [(s, pt, ot) if ot else (s, pt, o) for (s, pt, o, ot) in r]
        elif return_obj_text == "always":
            l = [(s, pt, ot) for (s, pt, o, ot) in r]
        else:
            l = [(s, pt, o) for (s, pt, o, ot) in r]
        return cast(List[Tuple[str, str, str]], l)

    ###############
    ###  Chats  ###
    ###############

    def add_chat(self, speaker_memid: str, chat: str) -> str:
        """Create a new chatNode

        Args:
            speaker_memid (string): memory ID of speaker
            chat (string): text representing chat
        """
        return ChatNode.create(self, speaker_memid, chat)

    def get_chat_by_id(self, memid: str) -> "ChatNode":
        """Return ChatNode, given memid

        Args:
            memid (string): Memory ID
        """
        return ChatNode(self, memid)

    def get_chat_id(self, speaker_id: str, chat: str) -> str:
        """Return memid of ChatNode, given speaker and chat

        Args:
            speaker_id: memid of speaker
            chat: chat string
        """
        r = self._db_read("SELECT uuid FROM Chats where speaker = ? and chat = ?", speaker_id, chat)
        return r[0][0]

    def get_recent_chats(self, n=1) -> List["ChatNode"]:
        """Return a list of at most n chats

        Args:
            n (int): number of recent chats
        """
        r = self._db_read("SELECT uuid FROM Chats ORDER BY time DESC LIMIT ?", n)
        return [ChatNode(self, m) for m, in reversed(r)]

    def get_most_recent_incoming_chat(self, after=-1) -> Optional["ChatNode"]:
        """Get the most recent chat that came in since 'after'

        Args:
            after (int): Marks the beginning of time window (from now)
        """
        r = self._db_read_one(
            """
            SELECT uuid
            FROM Chats
            WHERE speaker != ? AND time >= ?
            ORDER BY time DESC
            LIMIT 1
            """,
            self.self_memid,
            after,
        )
        if r:
            return ChatNode(self, r[0])
        else:
            return None

    ###################
    ## Logical form ###
    ###################

    def add_logical_form(self, logical_form: dict):
        """Create a new ProgramNode

        Args:
            logical_form: the semantic parser's output
        """
        return ProgramNode.create(self, logical_form)

    def get_logical_form_by_id(self, memid: str) -> "ProgramNode":
        """Return ProgramNode, given memid

        Args:
            memid (string): Memory ID
        """
        return ProgramNode(self, memid)

    #################
    ###  Players  ###
    #################

    # TODO consolidate anything using eid
    def get_player_by_eid(self, eid) -> Optional["PlayerNode"]:
        """Given eid, retrieve PlayerNode

        Args:
            eid (int): Entity ID
        """
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", eid)
        if r:
            return PlayerNode(self, r[0])
        else:
            return None

    # TODO get all if there are more than one?
    def get_player_by_name(self, name) -> Optional["PlayerNode"]:
        """Given player name, retrieve PlayerNode

        Args:
            name (string): Player name
        """
        r = self._db_read_one(
            'SELECT uuid FROM ReferenceObjects WHERE ref_type="player" AND name=?', name
        )
        #        r = self._db_read_one("SELECT uuid FROM Players WHERE name=?", name)
        if r:
            return PlayerNode(self, r[0])
        else:
            return None

    def get_players_tagged(self, *tags) -> List["PlayerNode"]:
        """Given a list of tags, retrieve all players with the tags

        Args:
            tags (list): list of tags
        """
        tags += ("PLAYER",)
        memids = set.intersection(*[set(self.get_memids_by_tag(t)) for t in tags])
        return [self.get_player_by_id(memid) for memid in memids]

    def get_player_by_id(self, memid) -> "PlayerNode":
        """Given memid, retrieve PlayerNode

        Args:
            memid (string): memory ID
        """
        return PlayerNode(self, memid)

    ###################
    ###  Locations  ###
    ###################

    def add_location(self, xyz: XYZ) -> str:
        """Create a new LocationNode

        Args:
            xyz (tuple): XYZ coordinates
        """
        return LocationNode.create(self, xyz)

    def get_location_by_id(self, memid: str) -> "LocationNode":
        """Given memory ID, retrieve LocationNode

        Args:
            memid (string): Memory ID
        """
        return LocationNode(self, memid)

    ###############
    ###  Times  ###
    ###############

    def add_time(self, t: int) -> str:
        """Create a new TimeNode

        Args:
            t (int): time value
        """
        return TimeNode.create(self, t)

    def get_time_by_id(self, memid: str) -> "TimeNode":
        """Given memid, retrieve TimeNode

        Args:
            memid (string): Memory ID
        """
        return TimeNode(self, memid)

    #    ###############
    #    ###  Sets   ###
    #    ###############
    #
    #    def add_set(self, memid_list):
    #        set_memid = SetNode.create(self)
    #        self.add_objs_to_set(set_memid, memid_list)
    #        return SetNode(self, set_memid)
    #
    #    def add_objs_to_set(self, set_memid, memid_list):
    #        for mid in memid_list:
    #            self.add_triple(mid, "set_member_", set_memid)

    ###############
    ###  Tasks  ###
    ###############

    # TORCH this
    def task_stack_push(
        self, task, parent_memid: str = None, chat_effect: bool = False
    ) -> "TaskNode":
        """Create a task object in memory, add triples and add to task stack

        Args:
            task (Task): The task to be pushed
            parent_memid (string): Memory ID of the task's parent
            chat_effect (bool): If the task was a result of a chat, add the triple.

        Returns:
            TaskNode: A TaskNode object

        Examples ::
            >>> task = Move(agent, {"target": pos_to_np([0, 0 , 0]), "approx" : 3})
            >>> parent_memid = '10517cc584844659907ccfa6161e9d32'
            >>> task_stack_push(task, parent_memid)
        """

        memid = TaskNode.create(self, task)

        # Relations
        if parent_memid:
            self.add_triple(subj=memid, pred_text="_has_parent_task", obj=parent_memid)
        if chat_effect:
            chat = self.get_most_recent_incoming_chat()
            assert chat is not None, "chat_effect=True with no incoming chats"
            self.add_triple(subj=chat.memid, pred_text="chat_effect_", obj=memid)

        # Return newly created object
        return TaskNode(self, memid)

    # TORCH this
    def task_stack_update_task(self, memid: str, task):
        """Update task in memory

        Args:
            memid (string): Memory ID
            task (Task): The task object

        Returns:
            int: Number of rows affected

        Examples ::
            >>> task = Move(agent, {"target": pos_to_np([0, 12, 0]), "approx" : 3})
            >>> memid = '10517cc584844659907ccfa6161e9d32'
            >>> task_stack_update_task(task, memid)
        """
        self.db_write("UPDATE Tasks SET pickled=? WHERE uuid=?", self.safe_pickle(task), memid)

    # TORCH this
    def task_stack_peek(self) -> Optional["TaskNode"]:
        """Return the top of task stack

        Returns:
            TaskNode: TaskNode object or None

        Examples ::
            >>> task_stack_peek()
        """
        r = self._db_read_one(
            """
            SELECT uuid
            FROM Tasks
            WHERE finished < 0 AND paused = 0 AND prio > 0
            ORDER BY created DESC
            LIMIT 1
            """
        )
        if r:
            return TaskNode(self, r[0])
        else:
            return None

    # TORCH this
    # TODO fold this into basic_search
    def task_stack_pop(self) -> Optional["TaskNode"]:
        """Return the 'TaskNode' of the stack head and mark finished

        Returns:
            TaskNode: An object of type TaskNode

        Examples ::
            >>> task_stack_pop()
        """
        mem = self.task_stack_peek()
        if mem is None:
            raise ValueError("Called task_stack_pop with empty stack")
        self.db_write("UPDATE Tasks SET finished=? WHERE uuid=?", self.get_time(), mem.memid)
        return mem

    def task_stack_pause(self) -> bool:
        """Pause the stack and return True iff anything was stopped

        Returns:
            int: Number of rows affected
        """
        return self.db_write("UPDATE Tasks SET paused=1 WHERE finished < 0") > 0

    def task_stack_clear(self):
        """Clear the task stack

        Returns:
            int: Number of rows affected
        """
        # FIXME use forget; fix this when tasks become MemoryNodes
        self.db_write("DELETE FROM Tasks WHERE finished < 0")

    def task_stack_resume(self) -> bool:
        """Resume stopped tasks. Return True if there was something to resume.

        Returns:
            int: Number of rows affected
        """
        return self.db_write("UPDATE Tasks SET paused=0") > 0

    def task_stack_find_lowest_instance(
        self, cls_names: Union[str, Sequence[str]]
    ) -> Optional["TaskNode"]:
        """Find and return the lowest item in the stack of the given class(es)

        Args:
            cls_names (Sequence): Class names of tasks

        Returns:
            TaskNode: A TaskNode object

        Examples ::
            >>> cls_names = 'Move'
            >>> task_stack_find_lowest_instance(cls_names)
        """
        names = [cls_names] if type(cls_names) == str else cls_names
        (memid,) = self._db_read_one(
            "SELECT uuid FROM Tasks WHERE {} ORDER BY created LIMIT 1".format(
                " OR ".join(["action_name=?" for _ in names])
            ),
            *names,
        )

        if memid is not None:
            return TaskNode(self, memid)
        else:
            return None

    def get_last_finished_root_task(self, action_name: str = None, recency: int = None):
        """Get last task that was marked as finished

        Args:
            action_name (string): Name of action associated with task
            recency (int): How recent should the task be

        Returns:
            TaskNode: A TaskNode object

        Examples ::
            >>> action_name = "BUILD"
            >>> get_last_finished_root_task (action_name=action_name)
        """
        q = """
        SELECT uuid
        FROM Tasks
        WHERE finished >= ? {}
        ORDER BY created DESC
        """.format(
            " AND action_name=?" if action_name else ""
        )
        if recency is None:
            recency = self.time.round_time(300)
        args: List = [self.get_time() - recency]
        if action_name:
            args.append(action_name)
        memids = [r[0] for r in self._db_read(q, *args)]
        for memid in memids:
            if self._db_read_one(
                "SELECT uuid FROM Triples WHERE pred_text='_has_parent_task' AND subj=?", memid
            ):
                # not a root task
                continue

            return TaskNode(self, memid)

    #########################
    ###  DialogueObjects  ###
    #########################

    # THIS SECTION IS TEMPORARY
    # FIXME!! agent
    # these are to be removed, and DialogueObjects merged with Tasks

    # FIXME agent
    def dialogue_stack_append_new(self, cls, *args, **kwargs):
        """Construct a new DialogueObject and append to stack"""
        self.dialogue_stack.stack.append(cls(*args, memory=self, **kwargs))

    #########################
    ###  Database Access  ###
    #########################

    def _db_read(self, query: str, *args) -> List[Tuple]:
        """Return all entries returned from running the query against
        the database.

        Args:
            query (string): The SQL query to be run against the database
            args: Arguments for the query

        Returns:
            list[tuple]: a list of tuples satisfying the query

        Examples::
            >>> query = "SELECT uuid FROM Memories WHERE node_type=?"
            >>> _db_read(query, 'Chat')
        """
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            r = c.fetchall()
            c.close()
            return r
        except:
            logging.error("Bad read: {} : {}".format(query, args))
            raise

    def _db_read_one(self, query: str, *args) -> Tuple:
        """Return one entry returned from running the query against
        the database

        Args:
            query (string): The query to be run against the database

        Returns:
            tuple: a single record or None

        Examples ::
            >>> query = "SELECT uuid FROM Memories WHERE node_type=?",
            >>> args = 'Chat'
            >>> _db_read_one(query, args)
        """
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            r = c.fetchone()
            c.close()
            return r
        except:
            logging.error("Bad read: {} : {}".format(query, args))
            raise

    def db_write(self, query: str, *args) -> int:
        """Return the number of rows affected.  As a side effect,
           sets the updated_time entry for each affected memory,
           and applies self.on_delete_callback to the list of deleted memids
           if there are any and on_delete_callback is not None

        Args:
            query (string): The query to be run against the database

        Returns:
            int: Number of rows affected

        Examples ::
            >>> query = "UPDATE Memories SET uuid=?"
            >>> args = '10517cc584844659907ccfa6161e9d32'
            >>> db_write(query, args)
        """
        r = self._db_write(query, *args)
        # some of this can be implemented with TRIGGERS and a python sqlite fn
        # but its a bit of a pain bc we want the agent's time in the update
        # not system time
        updated_memids = self._db_read("SELECT * FROM Updates")
        updated = [mem[0] for mem in updated_memids if mem[1] == "update"]
        deleted = [mem[0] for mem in updated_memids if mem[1] == "delete"]
        for u in set(updated):
            self.set_memory_updated_time(u)
        if self.on_delete_callback is not None and deleted:
            self.on_delete_callback(deleted)
        self._db_write("DELETE FROM Updates")
        # format the data to send to dashboard timeline
        query_table, query_operation = parse_sql(query[:query.find("(") - 1])
        query_dict = format_query(query, *args)
        hook_data = {
            "name" : "db_write", 
            "time" : self.get_time(), 
            "table_name" : query_table, 
            "operation" : query_operation, 
            "args" : query_dict, 
            "result" : r,
        }
        self._dispatch_signal.send(self.db_write, data=hook_data)
        return r

    def _db_write(self, query: str, *args) -> int:
        args = tuple(a.item() if isinstance(a, np.number) else a for a in args)
        try:
            c = self.db.cursor()
            c.execute(query, args)
            self.db.commit()
            c.close()
            self._write_to_db_log(query, *args)
            return c.rowcount
        except:
            logging.error("Bad write: {} : {}".format(query, args))
            raise

    def _db_script(self, script: str):
        """Execute a script against the database

        Args:
            script (string): the script to be run
        """
        c = self.db.cursor()
        c.executescript(script)
        self.db.commit()
        c.close()
        self._write_to_db_log(script, no_format=True)

    ####################
    ###  DB LOGGING  ###
    ####################

    def get_db_log_idx(self):
        """Return log index for database"""
        try:
            return self._db_log_idx
        except AttributeError:
            return None

    def _write_to_db_log(self, s: str, *args, no_format=False):
        """Write to database log file

        Args:
            s (string): query
            no_format (bool): no formatting needed
        """
        if not getattr(self, "_db_log_file", None):
            return

        # sub args in for ?
        split = s.split("?")
        final = b""
        for sub, arg in zip_longest(split, args, fillvalue=""):
            final += str(sub).encode("utf-8")
            if type(arg) == str and arg != "":
                # put quotes around string args
                final += '"{}"'.format(arg).encode("utf-8")
            else:
                final += str(arg).encode("utf-8")

        # remove newlines, add semicolon
        if not no_format:
            final = final.strip().replace(b"\n", b" ") + b";\n"

        # write to file
        self._db_log_file.write(final)
        self._db_log_file.flush()
        self._db_log_idx += 1

    ######################
    ###  MISC HELPERS  ###
    ######################

    def dump(self, sql_file, dict_memory_file=None):
        """Dump the database

        Args:
            sql_file (string): File to write database dump to
            dict_memory_file (string): File to dump task database to
        """
        sql_file.write("\n".join(self.db.iterdump()))
        if dict_memory_file is not None:
            import io

            assert type(dict_memory_file) == io.BufferedWriter
            dict_memory = {"task_db": self.task_db}
            pickle.dump(dict_memory, dict_memory_file)

    def reinstate_attrs(self, obj):
        """
        replace non-picklable attrs on blob data, using their values
        from the key-value store, indexed by the obj memid
        """
        for attr in NONPICKLE_ATTRS:
            if hasattr(obj, "__had_attr_" + attr):
                delattr(obj, "__had_attr_" + attr)
                setattr(obj, attr, self._safe_pickle_saved_attrs[obj.memid][attr])

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
                setattr(obj, "__had_attr_" + attr, True)
                self._safe_pickle_saved_attrs[obj.memid][attr] = val
        p = pickle.dumps(obj)
        self.reinstate_attrs(obj)
        return p

    def safe_unpickle(self, bs):
        """
        get non-picklable attrs from the key value store, and
        replace them on the blob data after retrieving from db
        """
        obj = pickle.loads(bs)
        self.reinstate_attrs(obj)
        return obj

    def register_hook(self, receiver, sender):
        """
        allows for registering hooks using the event dispatcher
        """
        allowed = [self.db_write,]
        if sender in allowed:
            self._dispatch_signal.connect(receiver, sender)
        else:
            raise ValueError("Unknown hook event {}. Available options are: {}".format(sender.__name__, allowed))
