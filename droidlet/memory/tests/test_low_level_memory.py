"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import unittest
import numpy as np
from droidlet.memory.memory_nodes import (
    SelfNode,
    PlayerNode,
    LocationNode,
    ChatNode,
    NamedAbstractionNode,
    TimeNode,
    TripleNode,
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
        # Test adding location
        loc_memid = self.memory.nodes[LocationNode.NODE_TYPE].create(self.memory, (10, 10, 10))

        # Test getting location by id
        assert self.memory.nodes[LocationNode.NODE_TYPE](self.memory, loc_memid).location == (
            10.0,
            10.0,
            10.0,
        )
        assert self.memory.nodes[LocationNode.NODE_TYPE](self.memory, loc_memid).pos == (
            10.0,
            10.0,
            10.0,
        )

    def test_time_apis(self):
        self.memory = AgentMemory()
        # Test adding time
        time_memid = self.memory.nodes[TimeNode.NODE_TYPE].create(self.memory, 10)

        # Test getting time by id
        assert self.memory.nodes[TimeNode.NODE_TYPE](self.memory, time_memid).time[0] == 10

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

        self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=sheep_memid, pred_text="has_property", obj=fluff_memid
        )
        self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=sheep_memid, pred_text="has_fur_color", obj_text="white"
        )
        assert (
            len(self.memory.nodes[TripleNode.NODE_TYPE].get_triples(self.memory, subj=sheep_memid))
            == 2
        )
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, subj=sheep_memid, pred_text="has_property"
                )
            )
            == 1
        )

    def test_tag(self):
        self.memory = AgentMemory()
        sheep_memid = NamedAbstractionNode.create(self.memory, "sheep")
        loc_memid = LocationNode.create(self.memory, (1, 2, 3))

        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=sheep_memid, tag_text="furry"
        )
        self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=sheep_memid, pred_text="has_home_location", obj=loc_memid
        )
        self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=sheep_memid, pred_text="has_fur_color", obj_text="white"
        )

        assert (
            len(self.memory.nodes[TripleNode.NODE_TYPE].get_triples(self.memory, subj=sheep_memid))
            == 3
        )
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, subj=sheep_memid, pred_text="has_home_location"
                )
            )
            == 1
        )

    def test_untag(self):
        self.memory = AgentMemory()
        player_memid = PlayerNode.create(
            self.memory, Player(10, "rachel", Pos(1, 0, 1), Look(0, 0))
        )

        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=player_memid, tag_text="girl"
        )
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=player_memid, tag_text="plays_football"
        )
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, subj=player_memid, obj_text="girl"
                )
            )
            == 1
        )
        self.memory.nodes[TripleNode.NODE_TYPE].untag(
            self.memory, subj_memid=player_memid, tag_text="girl"
        )
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, subj=player_memid, obj_text="girl"
                )
            )
            == 0
        )

    def test_memids_and_tags(self):
        self.memory = AgentMemory()
        player_memid = PlayerNode.create(
            self.memory, Player(10, "rache", Pos(1, 0, 1), Look(0, 0))
        )

        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=player_memid, tag_text="girl"
        )
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=player_memid, tag_text="plays_football"
        )

        # test get_memids_by_tag
        self.memory.nodes[TripleNode.NODE_TYPE].get_memids_by_tag(self.memory, tag="girl")
        assert (
            len(self.memory.nodes[TripleNode.NODE_TYPE].get_memids_by_tag(self.memory, tag="girl"))
            == 1
        )
        assert (
            self.memory.nodes[TripleNode.NODE_TYPE].get_memids_by_tag(self.memory, tag="girl")[0]
            == player_memid
        )

        # test_get_tags_by_memid
        assert "girl" in self.memory.nodes[TripleNode.NODE_TYPE].get_tags_by_memid(
            self.memory, player_memid
        )

        # test get_triples
        assert (
            len(
                self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
                    self.memory, subj=player_memid, obj_text="girl"
                )
            )
            == 1
        )

    # TODO: expand these
    def test_sql_form(self):
        self.memory = AgentMemory()
        self_mem = self.memory.get_mem_by_id(self.memory.self_memid)
        SelfNode.update(
            self.memory, Player(1, "robot", Pos(0, 0, 0), Look(0, 0)), self.memory.self_memid
        )
        rachel_memid = PlayerNode.create(
            self.memory, Player(10, "rachel", Pos(1, 0, 1), Look(0, 0))
        )

        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=rachel_memid, tag_text="girl"
        )
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=rachel_memid, tag_text="plays_football"
        )

        robert_memid = PlayerNode.create(
            self.memory, Player(11, "robert", Pos(4, 0, 5), Look(0, 0))
        )

        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=robert_memid, tag_text="boy"
        )
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=robert_memid, tag_text="plays_football"
        )

        sam_memid = PlayerNode.create(self.memory, Player(12, "sam", Pos(-2, 0, 5), Look(0, 0)))

        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=sam_memid, tag_text="girl"
        )
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=sam_memid, tag_text="plays_volleyball"
        )

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

        # test that text form and dict form return same records
        query_dict = {
            "output": "MEMORY",
            "memory_type": "ReferenceObject",
            "where_clause": {
                "OR": [
                    {"pred_text": "has_tag", "obj_text": "plays_volleyball"},
                    {"NOT": [{"pred_text": "has_tag", "obj_text": "girl"}]},
                ]
            },
        }

        memids_d, _ = m.search(self.memory, query=query_dict)
        assert set(memids_d) == set(memids)

        # test table property with tag
        m = MemorySearcher()
        query = "SELECT MEMORY FROM ReferenceObject WHERE ((has_tag=plays_volleyball) AND (x<0))"
        memids, _ = m.search(self.memory, query=query)
        assert robert_memid not in memids
        assert sam_memid in memids
        assert rachel_memid not in memids

        query = "SELECT (x, y) FROM ReferenceObject WHERE ((has_tag=plays_volleyball) AND (x<0))"
        memids, vals = m.search(self.memory, query=query)
        assert abs(vals[0][0] + 2.0) < 0.01
        assert abs(vals[0][1]) < 0.01

        triple_memid = TripleNode.create(
            self.memory, subj=sam_memid, pred_text="mother_of", obj=robert_memid
        )
        query = "SELECT MEMORY FROM ReferenceObject WHERE <<#{}, mother_of, ?>>".format(sam_memid)
        memids, _ = m.search(self.memory, query=query)
        assert robert_memid in memids
        assert len(memids) == 1

        # test FROM works
        query = "SELECT MEMORY FROM Triple WHERE create_time > -100"
        memids, _ = m.search(self.memory, query=query)
        assert all([type(self.memory.get_mem_by_id(m)) is TripleNode for m in memids])
        assert triple_memid in memids
        assert robert_memid not in memids

    def test_chat_apis_memory(self):
        self.memory = AgentMemory()
        # Test add_chat
        chat_memid = self.memory.nodes[ChatNode.NODE_TYPE].create(
            self.memory, speaker="463546548923408fdsgdsgfd", chat="are you around"
        )

        # test get_chat_by_id
        assert self.memory.nodes[ChatNode.NODE_TYPE](self.memory, chat_memid).TABLE == "Chats"

        # test get_recent_chats
        assert len(self.memory.nodes[ChatNode.NODE_TYPE].get_recent_chats(self.memory, n=5)) == 1
        _ = chat_memid = self.memory.nodes[ChatNode.NODE_TYPE].create(
            self.memory, speaker="fsfagfhgaft3764", chat="hello hello"
        )
        assert len(self.memory.nodes[ChatNode.NODE_TYPE].get_recent_chats(self.memory, n=5)) == 2

        # test get_most_recent_incoming_chat
        assert (
            self.memory.nodes[ChatNode.NODE_TYPE]
            .get_most_recent_incoming_chat(self.memory)
            .chat_text
            == "hello hello"
        )
        self.memory.forget(chat_memid)
        assert (
            self.memory.nodes[ChatNode.NODE_TYPE]
            .get_most_recent_incoming_chat(self.memory)
            .chat_text
            == "are you around"
        )

    def test_player_apis_memory(self):
        self.memory = AgentMemory()
        joe_memid = PlayerNode.create(self.memory, Player(10, "joey", Pos(1, 0, 1), Look(0, 0)))
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=joe_memid, tag_text="basketball_player"
        )
        ann_memid = PlayerNode.create(self.memory, Player(20, "ann", Pos(1, 0, 4), Look(0, 0)))
        self.memory.nodes[TripleNode.NODE_TYPE].tag(
            self.memory, subj_memid=ann_memid, tag_text="swimmer"
        )

        # Test get_player_by_eid
        assert (
            self.memory.nodes[PlayerNode.NODE_TYPE].get_player_by_eid(self.memory, 10).name
            == "joey"
        )

        # Test empty WHERE clause:
        _, memnodes = self.memory.basic_search("SELECT MEMORY FROM ReferenceObject")

        # Test basic_search to get player by name
        _, memnode = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE ref_type=player AND name=joey"
        )
        assert len(memnode) == 1
        assert memnode[0].eid == 10

        # Test basic_search to get player by tag
        p_id, p_nodes = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE ref_type=player AND has_tag=basketball_player"
        )
        assert len(p_id) == len(p_nodes) == 1
        assert p_nodes[0].eid == 10

        p_id, p_nodes = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE ref_type=player AND has_tag=swimmer"
        )
        assert len(p_id) == len(p_nodes) == 1
        assert p_nodes[0].eid == 20

        # Test getting player by id
        assert self.memory.nodes[PlayerNode.NODE_TYPE](self.memory, ann_memid).name == "ann"

    def test_triggers(self):
        self.memory = AgentMemory(agent_time=self.time)
        joe_memid = PlayerNode.create(self.memory, Player(10, "joe", Pos(1, 0, 1), Look(0, 0)))
        joe_tag_memid = self.memory.nodes[TripleNode.NODE_TYPE].tag(self.memory, joe_memid, "joe")
        jane_memid = PlayerNode.create(self.memory, Player(11, "jane", Pos(-1, 0, 1), Look(0, 0)))

        _, joe_mems = self.memory.basic_search("SELECT MEMORY FROM ReferenceObject WHERE name=joe")
        _, jane_mems = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE name=jane"
        )
        assert len(joe_mems) == 1
        assert len(jane_mems) == 1

        _, joe_mems_from_tag = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE has_tag=joe"
        )
        _, jane_mems_from_tag = self.memory.basic_search(
            "SELECT MEMORY FROM ReferenceObject WHERE has_tag=jane"
        )

        assert len(joe_mems_from_tag) == 1
        assert len(jane_mems_from_tag) == 0

        self.time.add_tick()
        brother_of_memid = self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=joe_memid, pred_text="brother_of", obj=jane_memid
        )
        sister_of_memid = self.memory.nodes[TripleNode.NODE_TYPE].create(
            self.memory, subj=jane_memid, pred_text="sister_of", obj=joe_memid
        )

        triples = self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
            self.memory, subj=jane_memid, pred_text="sister_of"
        )
        assert len(triples) == 1

        self.time.add_tick()
        self.memory.db_write("UPDATE ReferenceObjects SET x=? WHERE uuid=?", 2, joe_memid)
        cmd = "SELECT updated_time FROM Memories WHERE uuid=?"
        joe_t = self.memory._db_read(cmd, joe_memid)[0][0]
        jane_t = self.memory._db_read(cmd, jane_memid)[0][0]
        assert joe_t == 2
        assert jane_t == 0

        self.memory.forget(joe_memid)
        triples = self.memory.nodes[TripleNode.NODE_TYPE].get_triples(
            self.memory, subj=jane_memid, pred_text="sister_of"
        )
        assert len(triples) == 0


class PlaceFieldTest(unittest.TestCase):
    def test_place_field(self):
        memory = AgentMemory()
        PF = memory.place_field
        joe_x = 1
        joe_z = 2
        joe_loc = (joe_x, 0, joe_z)
        jane_loc = (-1, 0, 1)
        joe_memid = PlayerNode.create(memory, Player(10, "joe", Pos(*joe_loc), Look(0, 0)))
        jane_memid = PlayerNode.create(memory, Player(11, "jane", Pos(*jane_loc), Look(0, 0)))
        wall_locs = [{"pos": (-i, 0, 4)} for i in range(5)]
        changes = [{"pos": joe_loc, "memid": joe_memid}, {"pos": jane_loc, "memid": jane_memid}]
        changes.extend(wall_locs)
        PF.update_map(changes)
        assert PF.maps[0]["map"].sum() == 7
        jl = PF.memid2locs[joe_memid]
        assert len(jl) == 1
        recovered_pos = tuple(int(i) for i in PF.map2real(*PF.idx2ijh(list(jl.keys())[0])))
        assert recovered_pos == (joe_x, joe_z)
        assert len(PF.memid2locs["NULL"]) == 5
        changes = [{"pos": (-1, 0, 4), "is_delete": True}]
        PF.update_map(changes)
        assert len(PF.memid2locs["NULL"]) == 4
        new_jane_x = -5
        new_jane_z = 5
        changes = [{"pos": (new_jane_x, 0, new_jane_z), "memid": jane_memid, "is_move": True}]
        PF.update_map(changes)
        jl = PF.memid2locs[jane_memid]
        assert len(jl) == 1
        recovered_pos = tuple(int(i) for i in PF.map2real(*PF.idx2ijh(list(jl.keys())[0])))
        assert recovered_pos == (new_jane_x, new_jane_z)
        assert PF.maps[0]["map"].sum() == 6


if __name__ == "__main__":
    unittest.main()
