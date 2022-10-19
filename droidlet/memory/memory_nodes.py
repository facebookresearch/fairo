"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import uuid
import json
from typing import Optional, List, Dict, cast, Tuple
from droidlet.base_util import XYZ, POINT_AT_TARGET, to_player_struct


class MemoryNode:
    """This is the main class representing a node in the memory

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Examples::
        >>> node_list = [TaskNode, ChatNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> MemoryNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid"]
    PROPERTIES_BLACKLIST = ["agent_memory", "forgetme"]
    NODE_TYPE: Optional[str] = None

    @classmethod
    def new(cls, agent_memory, snapshot=False) -> str:
        """Creates a new entry into the Memories table

        Returns:
            string: memid of the entry
        """
        memid = uuid.uuid4().hex
        t = agent_memory.get_time()
        agent_memory.db_write(
            "INSERT INTO Memories VALUES (?,?,?,?,?,?)", memid, cls.NODE_TYPE, t, t, t, snapshot
        )
        return memid

    def __init__(self, agent_memory, memid: str):
        self.agent_memory = agent_memory
        self.memid = memid

    def get_tags(self) -> List[str]:
        return self.agent_memory.nodes[TripleNode.NODE_TYPE].get_tags_by_memid(
            self.agent_memory, self.memid
        )

    def get_properties(self) -> Dict[str, str]:
        blacklist = self.PROPERTIES_BLACKLIST + self._more_properties_blacklist()
        return {k: v for k, v in self.__dict__.items() if k not in blacklist}

    def update_recently_attended(self) -> None:
        self.agent_memory.set_memory_attended_time(self.memid)
        self.snapshot(self.agent_memory)

    def _more_properties_blacklist(self) -> List[str]:
        """Override in subclasses to add additional keys to the properties blacklist"""
        return []

    def snapshot(self, agent_memory):
        """Override in subclasses if necessary to properly snapshot."""

        read_cmd = "SELECT "
        for r in self.TABLE_COLUMNS:
            read_cmd += r + ", "
        read_cmd = read_cmd.strip(", ")
        read_cmd += " FROM " + self.TABLE + " WHERE uuid=?"
        data = agent_memory._db_read_one(read_cmd, self.memid)
        if not data:
            raise ("tried to snapshot nonexistent memory")

        archive_memid = self.new(agent_memory, snapshot=True)
        new_data = list(data)
        new_data[0] = archive_memid

        if hasattr(self, "ARCHIVE_TABLE"):
            archive_table = self.ARCHIVE_TABLE
        else:
            archive_table = self.TABLE
        write_cmd = "INSERT INTO " + archive_table + "("
        qs = ""
        for r in self.TABLE_COLUMNS:
            write_cmd += r + ", "
            qs += "?, "
        write_cmd = write_cmd.strip(", ")
        write_cmd += ") VALUES (" + qs.strip(", ") + ")"
        agent_memory.db_write(write_cmd, *new_data)
        link_archive_to_mem(agent_memory, self.memid, archive_memid)


def link_archive_to_mem(agent_memory, memid, archive_memid):
    agent_memory.nodes[TripleNode.NODE_TYPE].create(
        agent_memory, subj=archive_memid, pred_text="_archive_of", obj=memid
    )
    agent_memory.nodes[TripleNode.NODE_TYPE].create(
        agent_memory, subj=memid, pred_text="_has_archive", obj=archive_memid
    )


def dehydrate(lf):
    """
    replace any MemoryNode m in a logical form with {"dehydrated_mem": m.memid}
    This is used to store a logical form in the db; as logical forms may contain
    MemoryNodes as values, this makes it easier to serialize (text instead of python object).
    """
    for k, v in lf.items():
        if isinstance(v, MemoryNode):
            lf[k] = {"dehydrated_mem": v.memid}
        elif type(v) is list:
            for d in v:
                dehydrate(d)
        elif type(v) is dict:
            dehydrate(v)
        else:
            pass


class ProgramNode(MemoryNode):
    """This class represents the logical forms (outputs from
    the semantic parser)

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        logical_form (dict): The semantic parser output for text

    Examples::
        >>> node_list = [TaskNode, ChatNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> ProgramNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "logical_form"]
    TABLE = "Programs"
    NODE_TYPE = "Program"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        text = self.agent_memory._db_read_one(
            "SELECT logical_form FROM Programs WHERE uuid=?", self.memid
        )[0]
        lf = json.loads(text)
        self.rehydrate(lf)
        self.logical_form = lf

    @classmethod
    def create(cls, memory, logical_form: dict, snapshot=False) -> str:
        """Creates a new entry into the Programs table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> logical_form = {"dialogue_type" : "HUMAN_GIVE_COMMAND",
                                "action_sequence" : [
                                    {
                                        "action_type": "BUILD",
                                        "schematic": {"has_name": "sphere"},
                                    }]}
            >>> create(memory, logical_form)
        """
        memid = cls.new(memory, snapshot=snapshot)
        dehydrate(logical_form)
        logical_form_text = json.dumps(logical_form)
        memory.db_write(
            "INSERT INTO Programs(uuid, logical_form) VALUES (?,?)",
            memid,
            format(logical_form_text),
        )
        return memid

    def rehydrate(self, lf):
        """
        replace any {"dehydrated_mem": m.memid} with the associated MemoryNode
        This is used when retrieving a logical form in the db; as logical forms may contain
        MemoryNodes as values, this makes it easier to serialize (text instead of python object).
        """
        for k, v in lf.items():
            if type(v) is dict:
                memid = v.get("dehydrated_mem")
                if memid:
                    lf[k] = self.agent_memory.get_mem_by_id(memid)
                else:
                    self.rehydrate(v)
            elif type(v) is list:
                for d in v:
                    self.rehydrate(d)


# TODO FIXME instantiate as a side effect of making triples
class NamedAbstractionNode(MemoryNode):
    """This class represents an abstract concept with a name,
    to be used in triples

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        name (string): Name of the abstraction, for example : "has_tag"

    Examples::
        >>> node_list = [TaskNode, ChatNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> NamedAbstractionNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "name"]
    TABLE = "NamedAbstractions"
    NODE_TYPE = "NamedAbstraction"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        name = self.agent_memory._db_read_one(
            "SELECT name FROM NamedAbstractions WHERE uuid=?", self.memid
        )
        self.name = name

    @classmethod
    def create(cls, memory, name, snapshot=False) -> str:
        """Creates a new entry into the NamedAbstractions table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> name = "has_tag"
            >>> create(memory, name)
        """
        memid = memory._db_read_one("SELECT uuid FROM NamedAbstractions WHERE name=?", name)
        if memid:
            return memid[0]
        memid = cls.new(memory, snapshot=snapshot)
        memory.db_write("INSERT INTO NamedAbstractions(uuid, name) VALUES (?,?)", memid, name)
        return memid


class TripleNode(MemoryNode):
    """This class represents a (subject, predicate, object) KB triple

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        subj_text (string): the text of the subject
        subj (string):      the memid of the subject
        pred_text (string): the text of the predicate
        pred (string):      the memid of the predicate (a NamedAbstraction)
        obj_text (string):  the text of the object
        obj (string):       the memid of the object
        confidence (float): float between 0 and 1, currently unused

    """

    TABLE_COLUMNS = [
        "uuid",
        "subj",
        "subj_text",
        "pred",
        "pred_text",
        "obj",
        "obj_text",
        "confidence",
    ]
    TABLE = "Triples"
    NODE_TYPE = "Triple"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        self.triple = self.agent_memory._db_read_one(
            "SELECT subj_text, subj, pred_text, pred, obj_text, obj, confidence FROM Triples WHERE uuid=?",
            self.memid,
        )

    @classmethod
    def create(
        cls,
        memory,
        snapshot: bool = False,
        subj: str = None,  # this is a memid if given
        obj: str = None,  # this is a memid if given
        subj_text: str = None,
        pred_text: str = "has_tag",
        obj_text: str = None,
        confidence: float = 1.0,
    ) -> str:
        """Adds (subj, pred, obj) triple to the triplestore.
            *_text is the name field of a NamedAbstraction; if
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
            str: memid of triple


        """
        assert subj or subj_text
        assert obj or obj_text
        assert not (subj and subj_text)
        assert not (obj and obj_text)
        pred = NamedAbstractionNode.create(memory, pred_text)
        if not obj:
            obj = NamedAbstractionNode.create(memory, obj_text)
        if not subj:
            subj = NamedAbstractionNode.create(memory, subj_text)
        # check if triple exists, don't make it again:
        old_memids = memory._db_read(
            "SELECT uuid FROM Triples where pred=? and subj=? and obj=?", pred, subj, obj
        )
        if len(old_memids) > 0:
            # TODO error if more than 1
            return old_memids[0]

        memid = cls.new(memory, snapshot=snapshot)
        memory.db_write(
            "INSERT INTO Triples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            memid,
            subj,
            subj_text,
            pred,
            pred_text,
            obj,
            obj_text,
            confidence,
        )
        return memid

    # does not search archived mems for now
    # TODO clean up input?
    @classmethod
    def get_triples(
        self,
        agent_memory,
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
            >>> get_triples(agent_memory,
                            subj=subj,
                            pred_text=pred_text,
                            obj_text=obj_text)
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
        r = agent_memory._db_read(sql, *args)
        # subj is always returned as memid, even if pred and obj are returned as text
        # pred is always returned as text
        if return_obj_text == "if_exists":
            l = [(s, pt, ot) if ot else (s, pt, o) for (s, pt, o, ot) in r]
        elif return_obj_text == "always":
            l = [(s, pt, ot) for (s, pt, o, ot) in r]
        else:
            l = [(s, pt, o) for (s, pt, o, ot) in r]
        return cast(List[Tuple[str, str, str]], l)

    @classmethod
    def tag(self, agent_memory, subj_memid: str, tag_text: str):
        """Tag the subject with tag text.

        Args:
            subj_memid (string): memid of subject
            tag_text (string): string representation of the tag

        Returns:
            memid of triple representing the tag

        Examples::
            >>> subj_memid = '10517cc584844659907ccfa6161e9d32'
            >>> tag_text = "shiny"
            >>> tag(agent_memory, subj_memid, tag_text)
        """
        return agent_memory.nodes[TripleNode.NODE_TYPE].create(
            agent_memory, subj=subj_memid, pred_text="has_tag", obj_text=tag_text
        )

    @classmethod
    def get_tags_by_memid(
        self, agent_memory, subj_memid: str, return_text: bool = True
    ) -> List[str]:
        """Find all tag for a given memid

        Args:
            subj_memid (string): the subject's memid (uuid from Memories table)
            return_text (bool): if true, return the object text, otherwise return object memid

        Returns:
            list[string]: list of tags.

        Examples::
            >>> subj_memid = '10517cc584844659907ccfa6161e9d32'
            >>> get_tags_by_memid(agent_memory, subj_memid=subj_memid, return_text=True)
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
        r = agent_memory._db_read(q, subj_memid)
        return [x for (x,) in r]

    # does not search archived mems for now
    # assumes tag is tag text
    @classmethod
    def get_memids_by_tag(self, agent_memory, tag: str) -> List[str]:
        """Find all memids with a given tag

        Args:
            tag (string): string representation of the tag

        Returns:
            list[string]: list of memory ids (which are strings)

        Examples::
            >>> tag = "round"
            >>> get_memids_by_tag(agent_memory, tag)
        """
        r = agent_memory._db_read(
            'SELECT DISTINCT(Memories.uuid) FROM Memories INNER JOIN Triples as T ON T.subj=Memories.uuid WHERE T.pred_text="has_tag" AND T.obj_text=? AND Memories.is_snapshot=0',
            tag,
        )
        return [x for (x,) in r]

    @classmethod
    # TODO remove_triple
    def untag(self, agent_memory, subj_memid: str, tag_text: str):
        """Delete tag for subject

        Args:
            subj_memid (string): memid of subject
            tag_text (string): string representation of the tag

        Examples::
            >>> subj_memid = '10517cc584844659907ccfa6161e9d32'
            >>> tag_text = "shiny"
            >>> untag(agent_memory, subj_memid, tag_text)
        """
        # FIXME replace me with a basic filters when _self handled better
        triple_memids = agent_memory._db_read(
            'SELECT uuid FROM Triples WHERE subj=? AND pred_text="has_tag" AND obj_text=?',
            subj_memid,
            tag_text,
        )
        if triple_memids:
            agent_memory.forget(triple_memids[0][0])


class InterpreterNode(MemoryNode):
    """for representing interpreter objects"""

    TABLE_COLUMNS = ["uuid"]
    TABLE = "InterpreterMems"
    NODE_TYPE = "Interpreter"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        finished, awaiting_response, interpreter_type = agent_memory._db_read_one(
            "SELECT finished, awaiting_response, interpreter_type FROM InterpreterMems where uuid=?",
            memid,
        )
        self.finished = finished
        self.awaiting_response = awaiting_response
        self.interpreter_type = interpreter_type

    @classmethod
    def create(
        cls,
        memory,
        interpreter_type="interpeter",
        finished=False,
        awaiting_response=False,
        snapshot=False,
    ) -> str:
        memid = cls.new(memory, snapshot=snapshot)
        memory.db_write(
            "INSERT INTO InterpreterMems(uuid, finished, awaiting_response, interpreter_type) VALUES (?,?,?,?)",
            memid,
            finished,
            awaiting_response,
            interpreter_type,
        )
        return memid

    def finish(self):
        self.agent_memory.db_write(
            "UPDATE InterpreterMems SET finished=? WHERE uuid=?", 1, self.memid
        )


# the table entry just has the memid and a modification time,
# actual set elements are handled as triples
class SetNode(MemoryNode):
    """for representing sets of objects, so that it is easier to build complex relations
    using RDF/triplestore format."""

    TABLE_COLUMNS = ["uuid"]
    TABLE = "SetMems"
    NODE_TYPE = "Set"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)

    # FIXME put the member triples
    @classmethod
    def create(cls, memory, snapshot=False) -> str:
        memid = cls.new(memory, snapshot=snapshot)
        memory.db_write("INSERT INTO SetMems(uuid) VALUES (?)", memid)
        return memid

    def get_members(self):
        return self.agent_memory.nodes[TripleNode.NODE_TYPE].get_triples(
            self.agent_memory, pred_text="member_of", obj=self.memid
        )

    def snapshot(self, agent_memory):
        return SetNode.create(agent_memory, snapshot=True)


class ReferenceObjectNode(MemoryNode):
    """This is a class representing generic memory node for anything that has a spatial location and can be
    used a spatial reference (e.g. to the left of the x).

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Examples::
        >>> node_list = [TaskNode, ChatNode, ReferenceObjectNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> ReferenceObjectNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE = "ReferenceObjects"
    NODE_TYPE = "ReferenceObject"
    ARCHIVE_TABLE = "ArchivedReferenceObjects"

    def get_pos(self) -> XYZ:
        raise NotImplementedError("must be implemented in subclass")

    def get_point_at_target(self) -> POINT_AT_TARGET:
        raise NotImplementedError("must be implemented in subclass")

    def get_bounds(self):
        raise NotImplementedError("must be implemented in subclass")


class PlayerNode(ReferenceObjectNode):
    """This class represents humans and other agents that can affect
    the world

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        eid (int): Entity ID
        name (string): Name of player
        pos (tuple(float, float, float)): x, y, z coordinates
        pitch (float): the vertical angle of the agent's view vector
        yaw (float): the horizontal rotation angle of the agent's view vector

    Examples::
        >>> node_list = [TaskNode, ChatNode, PlayerNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> PlayerNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "eid", "name", "x", "y", "z", "pitch", "yaw", "ref_type"]
    NODE_TYPE = "Player"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        eid, name, x, y, z, pitch, yaw = self.agent_memory._db_read_one(
            "SELECT eid, name, x, y, z, pitch, yaw FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.eid = eid
        self.name = name
        self.pos = (x, y, z)
        self.pitch = pitch
        self.yaw = yaw

    @classmethod
    def create(cls, memory, player_struct, memid=None) -> str:
        """Creates a new entry into the ReferenceObjects table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> from collections import namedtuple
            >>> Player = namedtuple("Player", "entityId, name, pos, look, mainHand")
            >>> player_struct = Player(
                12345678, "dashboard", Pos(0.0, 0.0, 0.0), Look(0.0, 0.0), Item(0, 0)
            )
            >>> create(memory, player_struct)
        """
        memid = memid or cls.new(memory)

        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, name, x, y, z, pitch, yaw, ref_type) VALUES (?,?,?,?,?,?,?,?,?)",
            memid,
            player_struct.entityId,
            player_struct.name,
            player_struct.pos.x,
            player_struct.pos.y,
            player_struct.pos.z,
            player_struct.look.pitch,
            player_struct.look.yaw,
            "player",
        )
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_player")
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_physical_object")
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_animate")
        # this is a hack until memory_filters does "not"
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_not_location")

        if player_struct.name is not None:
            memory.nodes[TripleNode.NODE_TYPE].create(
                memory, subj=memid, pred_text="has_name", obj_text=player_struct.name
            )
        return memid

    @classmethod
    def update(cls, memory, p, memid) -> str:
        cmd = "UPDATE ReferenceObjects SET eid=?, name=?, x=?,  y=?, z=?, pitch=?, yaw=? WHERE "
        cmd = cmd + "uuid=?"
        memory.db_write(
            cmd, p.entityId, p.name, p.pos.x, p.pos.y, p.pos.z, p.look.pitch, p.look.yaw, memid
        )
        return memid

    def get_pos(self) -> XYZ:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos

    def get_yaw_pitch(self):
        yaw, pitch = self.agent_memory._db_read_one(
            "SELECT yaw, pitch FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.yaw = yaw
        self.pitch = pitch
        return yaw, pitch

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.pos
        # use the block above the player as point_at_target
        return cast(POINT_AT_TARGET, (x, y + 1, z, x, y + 1, z))

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z

    def get_struct(self):
        return to_player_struct(self.pos, self.yaw, self.pitch, self.eid, self.name)

    # TODO consolidate anything using eid
    @classmethod
    def get_player_by_eid(cls, agent_memory, eid) -> Optional["PlayerNode"]:
        """Given eid, retrieve PlayerNode

        Args:
            eid (int): Entity ID
        """
        r = agent_memory._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", eid)
        if r:
            return PlayerNode(agent_memory, r[0])
        else:
            return None


class SelfNode(PlayerNode):
    """This class is a special PlayerNode for representing the
    agent's self

    Args:
        agent_memory  (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Examples::
        >>> node_list = [TaskNode, ChatNode, PlayerNode, SelfNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> SelfNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "eid", "name", "x", "y", "z", "pitch", "yaw", "ref_type"]
    NODE_TYPE = "Self"

    @classmethod
    def create(cls, memory, player_struct=None, memid=None) -> str:
        """Creates a new entry into the ReferenceObjects table

        Returns:
            string: memid of the entry

        """
        memid = memid or cls.new(memory)
        if player_struct is None:
            eid, name, x, y, z, pitch, yaw = None, None, None, None, None, None, None
        else:
            eid, name = player_struct.entityId, player_struct.name
            x, y, z = player_struct.pos
            yaw, pitch = player_struct.look
        cmd = "INSERT INTO ReferenceObjects(uuid, eid, name, x, y, z, pitch, yaw, ref_type) VALUES (?,?,?,?,?,?,?,?,?)"
        memory.db_write(cmd, memid, eid, name, x, y, z, pitch, yaw, "self")
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "AGENT")
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "SELF")
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_physical_object")
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_animate")
        # this is a hack until memory_filters does "not"
        memory.nodes[TripleNode.NODE_TYPE].tag(memory, memid, "_not_location")

        if name is not None:
            memory.nodes[TripleNode.NODE_TYPE].create(
                memory, subj=memid, pred_text="has_name", obj_text=player_struct.name
            )
        return memid


# locations should always be archives?
class LocationNode(ReferenceObjectNode):
    """This is a ReferenceObjectNode representing a raw location
    (a point in space)

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        location (tuple): (x, y, z) coordinates of the node
        pos (tuple): (x, y, z) coordinates of the node

    Examples::
        >>> node_list = [TaskNode, ChatNode, LocationNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> LocationNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "x", "y", "z", "ref_type"]
    NODE_TYPE = "Location"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.location = (x, y, z)
        self.pos = (x, y, z)

    @classmethod
    def create(cls, memory, xyz: XYZ) -> str:
        """Creates a new entry into the ReferenceObjects table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> xyz = [0.0, 1.0, 3.0]
            >>> create(memory, xyz)
        """
        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, x, y, z, ref_type) VALUES (?, ?, ?, ?, ?)",
            memid,
            xyz[0],
            xyz[1],
            xyz[2],
            "location",
        )
        return memid

    def get_bounds(self):
        x, y, z = self.pos
        return x, x, y, y, z, z

    def get_pos(self) -> XYZ:
        return self.pos

    def get_point_at_target(self) -> POINT_AT_TARGET:
        x, y, z = self.pos
        return cast(POINT_AT_TARGET, (x, y, z, x, y, z))


# locations should always be archives?
class AttentionNode(LocationNode):
    """This is a ReferenceObjectNode representing spatial attention

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        attended (string): name of the node that is attending

    Examples::
        >>> node_list = [TaskNode, ChatNode, AttentionNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> AttentionNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "x", "y", "z", "type_name", "ref_type"]
    NODE_TYPE = "Attention"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        # we use the type_name field to store whose attention it is
        attender = self.agent_memory._db_read_one(
            "SELECT type_name FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.attender = attender

    @classmethod
    def create(cls, memory, xyz: XYZ, attender=None) -> str:
        """Creates a new entry into the ReferenceObjects table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> xyz = [0.0, 2.0, 50.0]
            >>> attender = 12345678 # entity ID of player
            >>> create(memory, xyz, attender)
        """
        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, x, y, z, type_name, ref_type) VALUES (?, ?, ?, ?, ?, ?)",
            memid,
            xyz[0],
            xyz[1],
            xyz[2],
            attender,
            "attention",
        )
        return memid


class TimeNode(MemoryNode):
    """This class represents a temporal 'location'

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        time (int): the value of time

    Examples::
        >>> node_list = [TaskNode, ChatNode, TimeNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> TimeNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "time"]
    TABLE = "Times"
    NODE_TYPE = "Time"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        t = self.agent_memory._db_read_one("SELECT time FROM Times WHERE uuid=?", self.memid)
        self.time = t

    @classmethod
    def create(cls, memory, time: int) -> str:
        """Creates a new entry into the Times table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> time = 1234
            >>> create(memory, time)
        """
        memid = cls.new(memory)
        memory.db_write("INSERT INTO Times(uuid, time) VALUES (?, ?)", memid, time)
        return memid


class ChatNode(MemoryNode):
    """This node represents a chat/utterance from another agent/human

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        speaker_id (string): The memid of the speaker who sent the chat
        chat_text (string): The chat string
        time (int): The time at which the chat was delivered

    Examples::
        >>> node_list = [TaskNode, ChatNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> ChatNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = ["uuid", "speaker", "chat", "time"]
    TABLE = "Chats"
    NODE_TYPE = "Chat"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        speaker, chat_text, time = self.agent_memory._db_read_one(
            "SELECT speaker, chat, time FROM Chats WHERE uuid=?", self.memid
        )
        self.speaker_id = speaker
        self.chat_text = chat_text
        self.time = time

    @classmethod
    def create(cls, memory, speaker: str, chat: str) -> str:
        """Creates a new entry into the Chats table

        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> speaker = "189fsfagf9382jfjash" #'name' of the Player giving the command
            >>> chat = "come here"
            >>> create(memory, speaker, chat)
        """
        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO Chats(uuid, speaker, chat, time) VALUES (?, ?, ?, ?)",
            memid,
            speaker,
            chat,
            memory.get_time(),
        )
        return memid

    @classmethod
    def get_most_recent_incoming_chat(self, agent_memory, after=-1) -> Optional["ChatNode"]:
        """Get the most recent chat that came in since 'after'

        Args:
            after (int): Marks the beginning of time window (from now)
        """
        r = agent_memory._db_read_one(
            """
            SELECT uuid
            FROM Chats
            WHERE speaker != ? AND time >= ?
            ORDER BY time DESC
            LIMIT 1
            """,
            agent_memory.self_memid,
            after,
        )
        if r:
            return ChatNode(agent_memory, r[0])
        else:
            return None

    @classmethod
    def get_recent_chats(self, agent_memory, n=1) -> List["ChatNode"]:
        """Return a list of at most n chats

        Args:
            n (int): number of recent chats
        """
        r = agent_memory._db_read("SELECT uuid FROM Chats ORDER BY time DESC LIMIT ?", n)
        return [ChatNode(agent_memory, m) for m, in reversed(r)]


class TaskNode(MemoryNode):
    """This node represents a task object that was placed on
    the agent's task_stack

    Args:
        agent_memory (AgentMemory): An AgentMemory object
        memid (string): Memory ID for this node

    Attributes:
        task (object): Name of the task
        created (int): Time at which it was created
        finished (int): Time at which it was finished
        action_name (string): The name of action that corresponds to this task

    Examples::
        >>> node_list = [TaskNode]
        >>> schema_path = [os.path.join(os.path.dirname(__file__), "memory_schema.sql")]
        >>> agent_memory = AgentMemory(db_file=":memory:",
                                       schema_paths=schema_path,
                                       db_log_path=None,
                                       nodelist=node_list)
        >>> memid = '10517cc584844659907ccfa6161e9d32'
        >>> TaskNode(agent_memory=agent_memory, memid=memid)
    """

    TABLE_COLUMNS = [
        "uuid",
        "action_name",
        "pickled",
        "prio",
        "running",
        "run_count",
        "paused",
        "created",
        "finished",
    ]
    TABLE = "Tasks"
    NODE_TYPE = "Task"
    EGG_PRIO = -3
    CHECK_PRIO = 0
    FINISHED_PRIO = -1

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        self.update_node()
        pickled, created, action_name = self.agent_memory._db_read_one(
            "SELECT pickled, created, action_name FROM Tasks WHERE uuid=?", memid
        )
        self.task = self.agent_memory.safe_unpickle(pickled)
        self.created = created
        # TODO changeme to just "name"
        self.action_name = action_name
        self.memory = agent_memory

    def update_node(self):
        prio, running, run_count, finished, paused = self.agent_memory._db_read_one(
            "SELECT prio, running, run_count, finished, paused FROM Tasks WHERE uuid=?", self.memid
        )
        self.prio = prio
        self.paused = paused
        self.run_count = run_count
        self.running = running
        self.finished = finished

    @classmethod
    def create(cls, memory, task) -> str:
        """Creates a new entry into the Tasks table

        the input task can be an instantiated Task
            or a dict with followng structure:
            {"class": TaskClass,
             "task_data": {...}}
            in this case, the TaskClass is the uninstantiated class
            and the agent will run update_task() when it is instantiated


        Returns:
            string: memid of the entry

        Examples::
            >>> memory = AgentMemory()
            >>> task = Task()
            >>> create(memory, task)
        """
        old_memid = getattr(task, "memid", None)
        if old_memid:
            return old_memid
        memid = cls.new(memory)
        if type(task) is dict:
            # this is an egg to be hatched by agent
            prio = task["task_data"].get("task_node_data", {}).get("prio", cls.EGG_PRIO)
            running = task["task_data"].get("task_node_data", {}).get("running", 0)
            run_count = task["task_data"].get("task_node_data", {}).get("run_count", 0)
            action_name = task["class"].__name__.lower()
        else:
            action_name = task.__class__.__name__.lower()
            prio = cls.CHECK_PRIO
            running = 0
            run_count = task.run_count
            task.memid = memid
        memory._db_write(
            "INSERT INTO Tasks (uuid, action_name, pickled, prio, running, run_count, created) VALUES (?,?,?,?,?,?,?)",
            memid,
            action_name,
            memory.safe_pickle(task),
            prio,
            running,
            run_count,
            memory.get_time(),
        )
        return memid

    def step(self, agent):
        self.task.step(agent)
        self.update_task()

    def update_task(self, task=None):
        task = task or self.task
        self.memory.db_write(
            "UPDATE Tasks SET run_count=?, pickled=? WHERE uuid=?",
            task.run_count,
            self.memory.safe_pickle(task),
            self.memid,
        )

    def update_condition(self, conditions):
        """
        conditions is a dict with keys in
        "init_condition", "terminate_condition"
        and values being Condition objects
        """
        for k, condition in conditions.items():
            setattr(self.task, k, condition)
        self.update_task()

    # FIXME names/constants for some specific prios
    # use this to update prio or running, don't do it directly on task or in db!!
    def get_update_status(self, status, force_db_update=True, force_task_update=True):
        """
        status is a dict with possible keys "prio", "running", "paused", "finished".

        prio > CHECK_PRIO  :  run me if possible, check my terminate condition
        prio = CHECK_PRIO  :  check my init_condition, run if true
        prio < CHECK_PRIO :  don't even check init_condition, I'm done or unhatched
        prio = EGG_PRIO :  I'm unhatched

        running = 1 :  task should be stepped if possible and not explicitly paused
        running = 0 :  task should not be stepped

        finished >  0 :  task has run and is complete, completed at the time indicated
        finished = -1 :  task has not completed

        paused = 1 : explicitly stopped by some other process; don't check any condtions and leave me alone
        paused = 0 : go on as normal

        this method updates these columns of the DB for each of the keys if have values
        if force_db_update is set, these will be updated with the relevant attr from self.task
        even if it is not in the status dict.
        if force_task_update is set, the information will go the other way,
        and whatever is in the dict will be put on the task.
        """
        status_out = {}
        for k in ["finished", "prio", "running", "paused"]:
            # update the task itself, hopefully don't need to do this when task objects are re-written as MemoryNode s
            if force_task_update:
                s = status.get(k)
                if s:
                    setattr(self.task, k, s)
            if k == "finished":
                if self.task.finished:
                    status_out[k] = self.agent_memory.get_time()
                    # warning: using the order of the iterator!
                    status["running"] = 0
                    status["prio"] = self.FINISHED_PRIO
                else:
                    status_out[k] = -1
            else:
                status_out[k] = (
                    status.get(k) if status.get(k) is not None else getattr(self.task, k, None)
                )
            if (status.get(k) is not None) or (force_db_update and status_out[k]):
                cmd = "UPDATE Tasks SET " + k + "=? WHERE uuid=?"
                self.agent_memory.db_write(cmd, status_out[k], self.memid)
        return status_out

    # FIXME! or torch me
    def propagate_status(self, status):
        # if the parent of a task is paused, propagate to children
        # if parent is currently paused and then unpaused, propagate to children
        pass

    def add_child_task(self, t, prio=CHECK_PRIO + 1):
        """Add (and by default activate) a child task, and pass along the id
        of the parent task (current task).  A task can only have one direct
        descendant any any given time.  To add a list of children use a ControlBlock

        Args:
            t: the task to be added.  a *Task* object, not a TaskNode
               agent: the agent running this task
            prio: default 1 (CHECK_PRIO + 1), set to 0 (CHECK_PRIO) if you want the child task added but not activated,
                  None if you want it added but its conditions left in charge
        """
        TaskMem = TaskNode(self.memory, t.memid)
        if prio is not None:
            # TODO mark that child task has been forcefully activated if it has non-trivial run_condition?
            TaskMem.get_update_status({"prio": prio})
        TripleNode.create(self.memory, subj=t.memid, pred_text="_has_parent_task", obj=self.memid)
        TripleNode.create(self.memory, obj=t.memid, pred_text="_has_child_task", subj=self.memid)

    def get_chat(self) -> Optional[ChatNode]:
        """Return the memory of the chat that caused this task's creation, or None"""
        triples = self.agent_memory.nodes[TripleNode.NODE_TYPE].get_triples(
            self.agent_memory, pred_text="chat_effect_", obj=self.memid
        )
        if triples:
            chat_id, _, _ = triples[0]
            return ChatNode(self.agent_memory, chat_id)
        else:
            return None

    def get_parent_task(self) -> Optional["TaskNode"]:
        """Return the 'TaskNode' of the parent task, or None"""
        triples = self.agent_memory.nodes[TripleNode.NODE_TYPE].get_triples(
            self.agent_memory, subj=self.memid, pred_text="_has_parent_task"
        )
        if len(triples) == 0:
            return None
        elif len(triples) == 1:
            _, _, parent_memid = triples[0]
            return TaskNode(self.agent_memory, parent_memid)
        else:
            raise AssertionError("Task {} has multiple parents: {}".format(self.memid, triples))

    def get_root_task(self) -> Optional["TaskNode"]:
        mem = self
        parent = self.get_parent_task()
        while parent is not None:
            mem = parent
            parent = mem.get_parent_task()
        return mem

    def get_child_tasks(self) -> List["TaskNode"]:
        """Return tasks that were spawned beause of this task"""
        r = self.agent_memory.nodes[TripleNode.NODE_TYPE].get_triples(
            self.agent_memory, pred_text="_has_parent_task", obj=self.memid
        )
        memids = [m for m, _, _ in r]
        return [TaskNode(self.agent_memory, m) for m in memids]

    def all_descendent_tasks(self, include_root=False) -> List["TaskNode"]:
        """Return a list of 'TaskNode' objects whose _has_parent_task root is this task
        If include_root is True, include this node in the list.
        Tasks are returned in the order they were finished.
        """
        descendents = []
        q = [self]
        while q:
            task = q.pop()
            children = task.get_child_tasks()
            descendents.extend(children)
            q.extend(children)
        if include_root:
            descendents.append(self)
        return sorted(descendents, key=lambda t: t.finished)

    def __repr__(self):
        return "<TaskNode: {}>".format(self.task)


# list of nodes to register in memory
NODELIST = [
    TaskNode,
    ChatNode,
    LocationNode,
    AttentionNode,
    TripleNode,
    InterpreterNode,
    SetNode,
    TimeNode,
    PlayerNode,
    SelfNode,
    ProgramNode,
    NamedAbstractionNode,
    ReferenceObjectNode,
]
