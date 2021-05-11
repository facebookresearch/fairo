import os
from droidlet.memory.sql_memory import AgentMemory
from grid.agent.grid_memory_nodes import BotNode

SCHEMAS = [os.path.join(os.path.dirname(__file__), "../../base_agent/base_memory_schema.sql")]


class GridMemory(AgentMemory):
    def __init__(self, db_file=":memory", schema_paths=SCHEMAS):
        super(GridMemory, self).__init__(db_file=db_file, schema_paths=SCHEMAS)

    def update_bot_info(self, bot):
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", bot["entityId"])
        if r:
            x, y, z = bot["pos"]
            name = bot["name"]
            print(f"[Memory INFO]: update bot [{name}] position: ({x}, {y}, {z})")
            self.db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE eid=?", x, y, z, bot["entityId"]
            )
            (memid,) = r
        else:
            memid = BotNode.create(self, bot)

    def delete_bot(self, eid):
        bot = self._db_read_one("SELECT type_name FROM ReferenceObjects WHERE eid=?", eid)
        if not bot:
            return
        print(f"[Memory INFO]: delete bot [{bot[0]}]")
        self.db_write("DELETE FROM ReferenceObjects WHERE eid=?", eid)

    def get_all_bot_eids(self):
        r = self._db_read("SELECT eid FROM ReferenceObjects WHERE ref_type=?", "bot")
        return r
