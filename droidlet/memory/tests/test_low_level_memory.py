"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
from droidlet.memory.memory_nodes import (
    SelfNode,
    PlayerNode,
    LocationNode,
    ChatNode,
    NamedAbstractionNode,
)
from droidlet.memory.sql_memory import AgentMemory
from droidlet.base_util import Pos, Look, Player
from droidlet.memory.memory_filters import MemorySearcher


class IncrementTime:
    def __init__(self):
        self.time = 0

    def get_time(self):
        return self.time

    def add_tick(self):
        self.time += 1


class BasicTest(unittest.TestCase):
    def setUp(self):
        self.time = IncrementTime()

    def test_get_recent_entities(self):
        self.memory = AgentMemory()
        joe_memid = PlayerNode.create(self.memory, Player(10, "joe", Pos(1, 0, 1), Look(0, 0)))
        players = self.memory.get_recent_entities(memtype="Player")
        assert len(players) == 1
        jane_memid = PlayerNode.create(self.memory, Player(11, "jane", Pos(-1, 0, 1), Look(0, 0)))
        players = self.memory.get_recent_entities(memtype="Player")
        assert len(players) == 2

    def test_get_node_from_memid(self):
        self.memory = AgentMemory()
        joe_memid = PlayerNode.create(self.memory, Player(10, "joe", Pos(1, 0, 1), Look(0, 0)))
        assert self.memory.get_node_from_memid(joe_memid) == "Player"
        loc_memid = LocationNode.create(self.memory, (0, 0, 0))
        assert self.memory.get_node_from_memid(loc_memid) == "Location"

    def test_location_apis(self):
        self.memory = AgentMemory()
        # Test add_location
        loc_memid = self.memory.add_location((10, 10, 10))

        # Test get_location_by_id
        assert self.memory.get_location_by_id(loc_memid).location == (10.0, 10.0, 10.0)
        assert self.memory.get_location_by_id(loc_memid).pos == (10.0, 10.0, 10.0)

    def test_time_apis(self):
        self.memory = AgentMemory()
        # Test add_time
        time_memid = self.memory.add_time(10)

        # Test get_time_by_id
        assert self.memory.get_time_by_id(time_memid).time[0] == 10

    def test_get_mem_by_id(self):
        self.memory = AgentMemory()
        kavya_memid = PlayerNode.create(self.memory, Player(10, "kavya", Pos(1, 0, 1), Look(0, 0)))
        assert self.memory.get_mem_by_id(kavya_memid).NODE_TYPE == "Player"

    def test_check_memid_exists(self):
        self.memory = AgentMemory()
        chat_memid = ChatNode.create(self.memory, speaker="dfghjk123445", chat="hi there!")
        loc_memid = LocationNode.create(self.memory, (0, 2, 0))
        assert self.memory.check_memid_exists(chat_memid, "Chats") == True
        assert self.memory.check_memid_exists(loc_memid, "ReferenceObjects") == True

    def test_forget(self):
        self.memory = AgentMemory()
        chat_memid = ChatNode.create(self.memory, speaker="dfghjk123445", chat="hi there!")
        self.memory.forget(chat_memid)
        loc_memid = LocationNode.create(self.memory, (0, 2, 0))
        assert self.memory.check_memid_exists(chat_memid, "Chats") == False
        assert self.memory.check_memid_exists(loc_memid, "ReferenceObjects") == True

    def test_forget_by_query(self):
        self.memory = AgentMemory()
        chat_memid = ChatNode.create(self.memory, speaker="dfghjk123445", chat="hellooooo!")
        assert self.memory.check_memid_exists(chat_memid, "Chats") == True
        query = "SELECT uuid from Chats where chat = 'hellooooo!'"
        self.memory.forget_by_query(query)
        assert self.memory.check_memid_exists(chat_memid, "Chats") == False

    def test_add_triple(self):
        self.memory = AgentMemory()
        sheep_memid = NamedAbstractionNode.create(self.memory, "sheep")
        fluff_memid = NamedAbstractionNode.create(self.memory, "fluffy")

        self.memory.add_triple(subj=sheep_memid, pred_text="has_property", obj=fluff_memid)
        self.memory.add_triple(subj=sheep_memid, pred_text="has_fur_color", obj_text="white")
        assert len(self.memory.get_triples(subj=sheep_memid)) == 2
        assert len(self.memory.get_triples(subj=sheep_memid, pred_text="has_property")) == 1

    def test_tag(self):
        self.memory = AgentMemory()
        sheep_memid = NamedAbstractionNode.create(self.memory, "sheep")
        loc_memid = LocationNode.create(self.memory, (1, 2, 3))

        self.memory.tag(subj_memid=sheep_memid, tag_text="furry")
        self.memory.add_triple(subj=sheep_memid, pred_text="has_home_location", obj=loc_memid)
        self.memory.add_triple(subj=sheep_memid, pred_text="has_fur_color", obj_text="white")

        assert len(self.memory.get_triples(subj=sheep_memid)) == 3
        assert len(self.memory.get_triples(subj=sheep_memid, pred_text="has_home_location")) == 1

    def test_untag(self):
        self.memory = AgentMemory()
        player_memid = PlayerNode.create(
            self.memory, Player(10, "rachel", Pos(1, 0, 1), Look(0, 0))
        )

        self.memory.tag(subj_memid=player_memid, tag_text="girl")
        self.memory.tag(subj_memid=player_memid, tag_text="plays_football")
        assert len(self.memory.get_triples(subj=player_memid, obj_text="girl")) == 1
        self.memory.untag(subj_memid=player_memid, tag_text="girl")
        assert len(self.memory.get_triples(subj=player_memid, obj_text="girl")) == 0

    def test_memids_and_tags(self):
        self.memory = AgentMemory()
        player_memid = PlayerNode.create(
            self.memory, Player(10, "rache", Pos(1, 0, 1), Look(0, 0))
        )

        self.memory.tag(subj_memid=player_memid, tag_text="girl")
        self.memory.tag(subj_memid=player_memid, tag_text="plays_football")

        # test get_memids_by_tag
        self.memory.get_memids_by_tag(tag="girl")
        assert len(self.memory.get_memids_by_tag(tag="girl")) == 1
        assert self.memory.get_memids_by_tag(tag="girl")[0] == player_memid

        # test_get_tags_by_memid
        assert "girl" in self.memory.get_tags_by_memid(player_memid)

        # test get_triples
        assert len(self.memory.get_triples(subj=player_memid, obj_text="girl")) == 1

    # TODO: expand these
    def test_sql_form(self):
        self.memory = AgentMemory()
        # FIXME? should this be in memory init?
        # FIXME!! in agents use SelfNode instead of PlayerNode
        self_memid = SelfNode.create(
            self.memory, Player(1, "robot", Pos(0, 0, 0), Look(0, 0)), memid=self.memory.self_memid
        )
        rachel_memid = PlayerNode.create(
            self.memory, Player(10, "rachel", Pos(1, 0, 1), Look(0, 0))
        )

        self.memory.tag(subj_memid=rachel_memid, tag_text="girl")
        self.memory.tag(subj_memid=rachel_memid, tag_text="plays_football")

        robert_memid = PlayerNode.create(
            self.memory, Player(11, "robert", Pos(4, 0, 5), Look(0, 0))
        )

        self.memory.tag(subj_memid=robert_memid, tag_text="boy")
        self.memory.tag(subj_memid=robert_memid, tag_text="plays_football")

        sam_memid = PlayerNode.create(self.memory, Player(12, "sam", Pos(-2, 0, 5), Look(0, 0)))

        self.memory.tag(subj_memid=sam_memid, tag_text="girl")
        self.memory.tag(subj_memid=sam_memid, tag_text="plays_volleyball")

        # test NOT
        m = MemorySearcher()
        query = "SELECT MEMORY FROM ReferenceObject WHERE (NOT has_tag=girl)"
        memids, _ = m.search(self.memory, query=query)
        assert robert_memid in memids
        assert sam_memid not in memids
        assert rachel_memid not in memids

        # test OR
        m = MemorySearcher()
        query = "SELECT MEMORY FROM ReferenceObject WHERE ((has_tag=plays_volleyball) OR (NOT has_tag=girl))"
        memids, _ = m.search(self.memory, query=query)
        assert robert_memid in memids
        assert sam_memid in memids
        assert rachel_memid not in memids

        # test table property with tag
        m = MemorySearcher()
        query = "SELECT MEMORY FROM ReferenceObject WHERE ((has_tag=plays_volleyball) AND (x<0))"
        memids, _ = m.search(self.memory, query=query)
        assert robert_memid not in memids
        assert sam_memid in memids
        assert rachel_memid not in memids

    def test_chat_apis_memory(self):
        self.memory = AgentMemory()
        # Test add_chat
        chat_memid = self.memory.add_chat(
            speaker_memid="463546548923408fdsgdsgfd", chat="are you around"
        )

        # test get_chat_by_id
        assert self.memory.get_chat_by_id(chat_memid).TABLE == "Chats"

        # test get_recent_chats
        assert len(self.memory.get_recent_chats(n=5)) == 1
        _ = chat_memid = self.memory.add_chat(speaker_memid="fsfagfhgaft3764", chat="hello hello")
        assert len(self.memory.get_recent_chats(n=5)) == 2

        # test get_most_recent_incoming_chat
        assert self.memory.get_most_recent_incoming_chat().chat_text == "hello hello"
        self.memory.forget(chat_memid)
        assert self.memory.get_most_recent_incoming_chat().chat_text == "are you around"

    def test_player_apis_memory(self):
        self.memory = AgentMemory()
        joe_memid = PlayerNode.create(self.memory, Player(10, "joey", Pos(1, 0, 1), Look(0, 0)))
        self.memory.tag(joe_memid, "basketball_player")
        ann_memid = PlayerNode.create(self.memory, Player(20, "ann", Pos(1, 0, 4), Look(0, 0)))
        self.memory.tag(ann_memid, "swimmer")

        # Test get_player_by_eid
        assert self.memory.get_player_by_eid(10).name == "joey"

        # Test get_player_by_name
        assert self.memory.get_player_by_name("joey").eid == 10

        # Test get_players_tagged

        # Test get_player_by_id
        assert self.memory.get_player_by_id(ann_memid).name == "ann"

    def test_triggers(self):
        self.memory = AgentMemory(agent_time=self.time)
        joe_memid = PlayerNode.create(self.memory, Player(10, "joe", Pos(1, 0, 1), Look(0, 0)))
        joe_tag_memid = self.memory.tag(joe_memid, "joe")
        jane_memid = PlayerNode.create(self.memory, Player(11, "jane", Pos(-1, 0, 1), Look(0, 0)))

        joe_mems = self.memory.basic_search({"base_exact": {"name": "joe"}, "triples": []})
        jane_mems = self.memory.basic_search({"base_exact": {"name": "jane"}, "triples": []})
        assert len(joe_mems) == 1
        assert len(jane_mems) == 1

        joe_mems_from_tag = self.memory.basic_search(
            {"base_exact": {}, "triples": [{"obj_text": "joe", "pred_text": "has_tag"}]}
        )
        jane_mems_from_tag = self.memory.basic_search(
            {"base_exact": {}, "triples": [{"obj_text": "jane", "pred_text": "has_tag"}]}
        )
        assert len(joe_mems_from_tag) == 1
        assert len(jane_mems_from_tag) == 0

        self.time.add_tick()
        brother_of_memid = self.memory.add_triple(
            subj=joe_memid, pred_text="brother_of", obj=jane_memid
        )
        sister_of_memid = self.memory.add_triple(
            subj=jane_memid, pred_text="sister_of", obj=joe_memid
        )

        triples = self.memory.get_triples(subj=jane_memid, pred_text="sister_of")
        assert len(triples) == 1

        self.time.add_tick()
        self.memory.db_write("UPDATE ReferenceObjects SET x=? WHERE uuid=?", 2, joe_memid)
        cmd = "SELECT updated_time FROM Memories WHERE uuid=?"
        joe_t = self.memory._db_read(cmd, joe_memid)[0][0]
        jane_t = self.memory._db_read(cmd, jane_memid)[0][0]
        assert joe_t == 2
        assert jane_t == 0

        self.memory.forget(joe_memid)
        triples = self.memory.get_triples(subj=jane_memid, pred_text="sister_of")
        assert len(triples) == 0


if __name__ == "__main__":
    unittest.main()
