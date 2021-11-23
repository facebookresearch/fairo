"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.base_util import XYZ, Pos
from droidlet.memory.memory_nodes import ReferenceObjectNode, MemoryNode, NODELIST
import pickle


class DetectedObjectNode(ReferenceObjectNode):
    """Encapsulates all methods for dealing with object detections - creating / updating /
    retrieving them etc.

    Args:
        agent_memory (AgentMemory): reference to the agent's memory
        memid (string): memory id to create the DetectedObject for
    """

    TABLE = "ReferenceObjects"
    NODE_TYPE = "DetectedObject"
    TABLE_COLUMNS = ["uuid", "eid", "x", "y", "z", "ref_type"]

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        obj_id, x, y, z = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z FROM ReferenceObjects WHERE uuid=?", memid
        )
        self.obj_id = obj_id
        self.eid = obj_id
        self.pos = (x, y, z)

    def __repr__(self):
        return "DetectedObject id {}, pos {}".format(self.obj_id, self.pos)

    @classmethod
    def create(cls, memory, detected_obj) -> str:
        memid = cls.new(memory)
        bounds = detected_obj.get_bounds()
        memory.db_write(
            "INSERT INTO ReferenceObjects \
            (uuid, eid, x, y, z, ref_type) \
            VALUES (?, ?, ?, ?, ?, ?)",
            memid,
            detected_obj.eid,
            detected_obj.get_xyz()["x"],
            detected_obj.get_xyz()["y"],
            detected_obj.get_xyz()["z"],
            cls.NODE_TYPE,
        )
        memory.db_write(
            "INSERT INTO DetectedObjectFeatures(uuid, featureBlob, minx, miny, minz, maxx, maxy, maxz, bbox, mask) \
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            memid,
            pickle.dumps(detected_obj.feature_repr),
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            bounds[4],
            bounds[5],
            pickle.dumps(detected_obj.bbox),
            pickle.dumps(detected_obj.mask)
        )

        cls.safe_tag(detected_obj, memory, memid, "has_name", "label")
        cls.safe_tag(detected_obj, memory, memid, "has_colour", "color")
        if hasattr(detected_obj, "properties") and detected_obj.properties is not None:
            cls.safe_tag(detected_obj, memory, memid, "has_properties", "properties")
            for prop in detected_obj.properties:
                memory.tag(memid, prop)

        # Tag everything with has_tag predicate
        if hasattr(detected_obj, "color") and detected_obj.color is not None:
            memory.tag(memid, detected_obj.color)
        if hasattr(detected_obj, "label") and detected_obj.label is not None:
            memory.tag(memid, detected_obj.label)
        memory.tag(memid, "_physical_object")
        memory.tag(memid, "_not_location")
        return memid

    @classmethod
    def update(cls, memory, detected_obj) -> str:
        memids = memory._db_read(
            "SELECT uuid FROM ReferenceObjects WHERE eid=?", str(detected_obj.eid)
        )

        memid = memids[0][0]
        memory.set_memory_attended_time(memid)
        memory.db_write(
            "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?",
            detected_obj.get_xyz()["x"],
            detected_obj.get_xyz()["y"],
            detected_obj.get_xyz()["z"],
            memid,
        )

        # TODO: should we update mask, bbox and bounds too?
        memory.db_write(
            "UPDATE DetectedObjectFeatures SET featureBlob=? where uuid=?",
            pickle.dumps(detected_obj.feature_repr),
            memid,
        )
        return memid

    @classmethod
    def safe_tag(cls, detected_obj, memory, subj_memid, pred_text, attr):
        attr = getattr(detected_obj, attr, None)
        if attr is not None:
            memory.add_triple(subj=subj_memid, pred_text=pred_text, obj_text=str(attr))

    @classmethod
    def get_all(cls, memory) -> list:
        objs = []
        detected_objs = memory._db_read(
            "SELECT uuid, eid, x, y, z FROM ReferenceObjects WHERE ref_type=?", cls.NODE_TYPE
        )
        for node in detected_objs:
            objs.append(cls.from_node(memory, node))
        return objs

    @classmethod
    def from_node(cls, memory, node) -> list:
        def get_value(memid, pred_text):
            triple = memory.get_triples(
                subj=memid, pred_text=pred_text, return_obj_text="if_exists"
            )
            if len(triple) > 0 and len(triple[0]) >= 3 and triple[0][2] != memid:
                return triple[0][2]
            else:
                return None

        label = get_value(node[0], "has_name")
        color = get_value(node[0], "has_colour")
        properties = get_value(node[0], "has_properties")

        # Get DetectedObjectFeatures
        feature_blob, minx, miny, minz, maxx, maxy, maxz, bbox, mask = memory._db_read(
            "SELECT featureBlob, minx, miny, minz, maxx, maxy, maxz, bbox, mask \
                FROM DetectedObjectFeatures WHERE uuid=?", node[0]
        )[0]
        feature_repr = pickle.loads(feature_blob)
        bbox = pickle.loads(bbox)
        mask = pickle.loads(mask)

        return {
            "eid": node[1],
            "xyz": (node[2], node[3], node[4]),
            "label": label,
            "color": color,
            "properties": properties,
            "feature_repr": feature_repr,
            "bounds": (minx, miny, minz, maxx, maxy, maxz),
            "bbox": bbox,
            "mask": mask
        }

    def get_pos(self) -> XYZ:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos

    def get_bounds(self):
        minx, miny, minz, maxx, maxy, maxz = self.agent_memory._db_read_one(
            "SELECT minx, miny, minz, maxx, maxy, maxz FROM DetectedObjectFeatures WHERE uuid=?", self.memid
        )
        return (minx, miny, minz, maxx, maxy, maxz)

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self):
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        return Pos(x, y, z)


class HumanPoseNode(ReferenceObjectNode):
    """Encapsulates all methods for dealing with human poses - creating / updating /
    retrieving them etc.

    Args:
        agent_memory (AgentMemory): reference to the agent's memory
        memid (string): memory id to create the HumanPose for
    """

    TABLE = "ReferenceObjects"
    NODE_TYPE = "HumanPose"
    TABLE_COLUMNS = ["uuid", "eid", "x", "y", "z", "ref_type"]

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        obj_id, x, y, z = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z FROM ReferenceObjects WHERE uuid=?", memid
        )
        self.obj_id = obj_id
        self.eid = obj_id
        self.pos = (x, y, z)

    def __repr__(self):
        return "HumanPose id {}, pos {}".format(self.obj_id, self.pos)

    @classmethod
    def create(cls, memory, humanpose) -> str:
        memids = memory._db_read(
            "SELECT uuid FROM ReferenceObjects WHERE eid=?", str(humanpose.eid)
        )

        if len(memids) > 0:
            memory.set_memory_attended_time(memids[0])
            return memids[0]

        memid = cls.new(memory)
        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, x, y, z, ref_type) VALUES (?, ?, ?, ?, ?, ?)",
            memid,
            humanpose.eid,
            humanpose.xyz[0],
            humanpose.xyz[1],
            humanpose.xyz[2],
            cls.NODE_TYPE,
        )
        memory.db_write(
            "INSERT INTO HumanPoseFeatures(uuid, keypointsBlob) VALUES (?, ?)",
            memid,
            pickle.dumps(humanpose.keypoints),
        )
        memory.tag(memid, "_human_pose")
        return memid

    @classmethod
    def get_all(cls, memory) -> str:
        objs = []
        human_poses = memory._db_read(
            "SELECT uuid, eid, x, y, z FROM ReferenceObjects WHERE ref_type=?", cls.NODE_TYPE
        )
        for x in human_poses:
            # get feature blob
            feature_blob = memory._db_read(
                "SELECT keypointsBlob FROM HumanPoseFeatures WHERE uuid=?", x[0]
            )
            feature_repr = pickle.loads(feature_blob[0][0])
            objs.append({"eid": x[1], "xyz": (x[2], x[3], x[4]), "keypoints": feature_repr})
        return objs

    def get_pos(self) -> XYZ:
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos

    # TODO: use a smarter way to get point_at_target
    def get_point_at_target(self):
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        return Pos(x, y, z)


class DanceNode(MemoryNode):
    """Encapsulates all methods for dealing with dances - only creating them for now.

    Args:
        agent_memory (AgentMemory): reference to the agent's memory
        memid (string): memory id to retrieve the dance from
    """

    TABLE_COLUMNS = ["uuid"]
    TABLE = "Dances"
    NODE_TYPE = "Dance"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        # TODO put in DB/pickle like tasks?
        self.dance_fn = self.agent_memory.dances[memid]

    @classmethod
    def create(cls, memory, dance_fn, name=None, tags=[]) -> str:
        memid = cls.new(memory)
        memory.db_write("INSERT INTO Dances(uuid) VALUES (?)", memid)
        # TODO put in db via pickle like tasks?
        memory.dances[memid] = dance_fn
        if name is not None:
            memory.add_triple(subj=memid, pred_text="has_name", obj_text=name)
        if len(tags) > 0:
            for tag in tags:
                memory.add_triple(subj=memid, pred_text="has_tag", obj_text=tag)
        return memid


NODELIST = NODELIST + [DetectedObjectNode, HumanPoseNode, DanceNode]  # noqa
