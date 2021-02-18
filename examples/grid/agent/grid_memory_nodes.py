from base_agent.memory_nodes import ReferenceObjectNode


class BotNode(ReferenceObjectNode):
    """ a memory node represneting a bot"""

    TABLE_COLUMNS = ["uuid", "eid", "x", "y", "z", "ref_type", "type_name"]
    TABLE = "ReferenceObjects"
    NODE_TYPE = "Bot"

    def __init__(self, agent_memory, memid: str):
        super().__init__(agent_memory, memid)
        eid, x, y, z, yaw, pitch = self.agent_memory._db_read_one(
            "SELECT eid, x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.eid = eid
        self.pos = (x, y, z)

    @classmethod
    def create(cls, memory, bot):
        # TODO warn/error if bot already in memory?
        memid = cls.new(memory)
        x, y, z = bot["pos"]
        name = bot["name"]
        print(f"[Memory INFO]: Insert Bot [{name}] into memory, position: ({x}, {y}, {z})")
        memory.db_write(
            "INSERT INTO ReferenceObjects(uuid, eid, x, y, z, ref_type, type_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
            memid,
            bot["entityId"],
            x,
            y,
            z,
            "bot",
            bot["name"],
        )
        return memid

    def get_pos(self):
        x, y, z = self.agent_memory._db_read_one(
            "SELECT x, y, z FROM ReferenceObjects WHERE uuid=?", self.memid
        )
        self.pos = (x, y, z)
        return self.pos
