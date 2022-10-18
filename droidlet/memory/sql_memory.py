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
import datetime
from itertools import zip_longest
from typing import cast, Optional, List, Tuple, Sequence, Union
from droidlet.base_util import XYZ
from droidlet.shared_data_structs import Time
from droidlet.memory.memory_filters import MemorySearcher
from droidlet.event import dispatch
from droidlet.memory.memory_util import parse_sql, format_query
from droidlet.memory.place_field import PlaceField, EmptyPlaceField

from droidlet.memory.memory_nodes import (  # noqa
    TaskNode,
    TripleNode,
    SelfNode,
    PlayerNode,
    ProgramNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    ReferenceObjectNode,
    NamedAbstractionNode,
    AttentionNode,
    NODELIST,
)

# FIXME set these in the Task classes
NONPICKLE_ATTRS = [
    "agent",
    "memory",
    "agent_memory",
    "task_fns",
    "init_condition",
    "terminate_condition",
    "movement",
]

DEFAULT_PIXELS_PER_UNIT = 100
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
        searcher (MemorySearcher): A class to process searches through memory
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
        place_field_pixels_per_unit=DEFAULT_PIXELS_PER_UNIT,
    ):
        if db_log_path:
            self._db_log_file = gzip.open(db_log_path + ".gz", "w")
            self._db_log_idx = 0
        if os.path.isfile(db_file):
            os.remove(db_file)
        self.db = sqlite3.connect(db_file, check_same_thread=False)
        self.task_db = {}
        self._safe_pickle_saved_attrs = {}

        self.on_delete_callback = on_delete_callback

        self.init_time_interface(agent_time)

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
        # FIXME, this is ugly.  using for memtype/FROM clauses in searches
        # we also store .TABLE in each node, and use it.  this also should be fixed,
        # it breaks the abstraction
        self.node_children = {}
        for node in nodelist:
            self.node_children[node.NODE_TYPE] = []
            for possible_child in nodelist:
                if node in possible_child.__mro__:
                    self.node_children[node.NODE_TYPE].append(possible_child.NODE_TYPE)

        self.make_self_mem()

        self.searcher = MemorySearcher()
        if place_field_pixels_per_unit > 0:
            self.place_field = PlaceField(self, pixels_per_unit=place_field_pixels_per_unit)
        else:
            self.place_field = EmptyPlaceField()

    def __del__(self):
        """Close the database file"""
        if getattr(self, "_db_log_file", None):
            self._db_log_file.close()

    def make_self_mem(self):
        # create a "self" memory to reference in Triples
        self.self_memid = "0" * len(uuid.uuid4().hex)
        self.db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)", self.self_memid, "Self", 0, 0, -1, False
        )
        player_struct = None
        SelfNode.create(self, player_struct, memid=self.self_memid)
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "AGENT")
        self.nodes[TripleNode.NODE_TYPE].tag(self, self.self_memid, "SELF")

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

    def update(self):
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
        # FIXME what if memid doesn't exist?  what if mem was deleted?
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
        et"""
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

    def basic_search(self, query, get_all=False):
        """Perform a basic search using the query

        Args:
            query (dict): A FILTERS dict or sqly query

        Returns:
            list[memid], list[value]: the memids and respective values from the search

        """
        return self.searcher.search(self, query=query, get_all=get_all)

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
            self.nodes[TripleNode.NODE_TYPE].create(
                self, subj=memid, pred_text="_has_parent_task", obj=parent_memid
            )
        if chat_effect:
            chat = self.nodes[ChatNode.NODE_TYPE].get_most_recent_incoming_chat(self)
            assert chat is not None, "chat_effect=True with no incoming chats"
            self.nodes[TripleNode.NODE_TYPE].create(
                self, subj=chat.memid, pred_text="chat_effect_", obj=memid
            )

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

    def get_last_finished_root_task(
        self, action_name: str = None, recency: int = None, ignore_control: bool = False
    ):
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
        args: List = [max(self.get_time() - recency, 0)]
        if action_name:
            args.append(action_name)
        task_memids_ok = {r[0]: False for r in self._db_read(q, *args)}
        for memid in task_memids_ok:
            tmems = self._db_read_one(
                "SELECT obj FROM Triples WHERE pred_text='_has_parent_task' AND subj=?", memid
            )
            if tmems:  # not a root task
                if ignore_control:
                    is_parent_control = self.get_mem_by_id(tmems[0]).action_name == "controlblock"
                    is_self_control = self.get_mem_by_id(memid).action_name == "controlblock"
                    if is_parent_control and not is_self_control:
                        task_memids_ok[memid] = True
            else:  # no parent, this is a root task
                task_memids_ok[memid] = True
        ok_memids = [m for m, b in task_memids_ok.items() if b]
        if ok_memids:
            return TaskNode(self, ok_memids[0])
        else:
            return None

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
        start_time = datetime.datetime.now()
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
        query_table, query_operation = parse_sql(query[: query.find("(") - 1])
        query_dict = format_query(query, *args)
        # data is sent to the dashboard as JSON to be displayed in the timeline
        end_time = datetime.datetime.now()
        hook_data = {
            "name": "memory",
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_time": (end_time - start_time).total_seconds(),
            "agent_time": self.get_time(),
            "table_name": query_table,
            "operation": query_operation,
            "arguments": query_dict,
            "result": r,
        }
        dispatch.send("memory", data=hook_data)
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
