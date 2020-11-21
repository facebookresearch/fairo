"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import List
import torch

SELFID = "0" * 32


def maybe_and(sql, a):
    if a:
        return sql + " AND "
    else:
        return sql


def maybe_or(sql, a):
    if a:
        return sql + " OR "
    else:
        return sql


# FIXME merge this with more general filters
# FIXME don't just pick the first one, allow all props returned
def get_property_value(agent_memory, mem, prop, get_all=False):
    """
    Tries to get property value from a memory.
    
    Args:
        agent_memory: an AgentMemory object
        mem: a MemoryNode object
        prop: a string with the name of the property

    looks with the following order of precedence:
    1: main memory table
    2: table corresponding to the nodes .TABLE
    3: triple with the nodes memid as subject and prop as predicate
    """

    # is it in the main memory table?
    cols = [c[1] for c in agent_memory._db_read("PRAGMA table_info(Memories)")]
    if prop in cols:
        cmd = "SELECT " + prop + " FROM Memories WHERE uuid=?"
        r = agent_memory._db_read(cmd, mem.memid)
        return r[0][0]
    # is it in the mem.TABLE?
    T = mem.TABLE
    cols = [c[1] for c in agent_memory._db_read("PRAGMA table_info({})".format(T))]
    if prop in cols:
        cmd = "SELECT " + prop + " FROM " + T + " WHERE uuid=?"
        r = agent_memory._db_read(cmd, mem.memid)
        return r[0][0]
    # is it a triple?
    triples = agent_memory.get_triples(subj=mem.memid, pred_text=prop, return_obj_text="always")
    if len(triples) > 0:
        if get_all:
            return [t[2] for t in triples]
        else:
            return triples[0][2]

    return None


# TODO?  merge into Memory
class BasicMemorySearcher:
    def __init__(self, self_memid=SELFID, search_data=None):
        self.self_memid = self_memid
        self.search_data = search_data

    def is_filter_empty(self, search_data):
        r = search_data.get("special")
        if r and len(r) > 0:
            return False
        r = search_data.get("base_range")
        if r and len(r) > 0:
            return False
        r = search_data.get("base_exact")
        if r and len(r) > 0:
            return False
        r = search_data.get("memories_range")
        if r and len(r) > 0:
            return False
        r = search_data.get("memories_exact")
        if r and len(r) > 0:
            return False
        t = search_data.get("triples")
        if t and len(t) > 0:
            return False
        return True

    def range_queries(self, r, table, a=False):
        """ this does x, y, z, pitch, yaw, etc.
        input format for generates is 
        {"xmin": float, xmax: float, ... , yawmin: float, yawmax: float}
        """
        sql = ""
        vals = []
        for k, v in r.items():
            if "min" in k:
                sql = maybe_and(sql, len(vals) > 0)
                sql += table + "." + k.replace("min", "") + ">? "
                vals.append(v)
            if "max" in k:
                sql = maybe_and(sql, len(vals) > 0)
                sql += table + "." + k.replace("max", "") + "<? "
                vals.append(v)
        return sql, vals

    def exact_matches(self, m, table, a=False):
        sql = ""
        vals = []
        for k, v in m.items():
            sql = maybe_and(sql, len(vals) > 0)
            sql += table + "." + k + "=? "
            vals.append(v)
        return sql, vals

    def triples(self, triples, table, a=False):
        # currently does an "and": the memory needs to satisfy all triples
        vals = []
        if not triples:
            return "", vals
        sql = "{}.uuid IN (SELECT subj FROM Triples WHERE ".format(table)
        for t in triples:
            sql = maybe_or(sql, len(vals) > 0)
            if t.get("pred_text"):
                vals.append(t["pred_text"])
                if t.get("obj_text"):
                    sql += "(pred_text, obj_text)=(?, ?)"
                    vals.append(t["obj_text"])
                else:
                    sql += "(pred_text, obj)=(?, ?)"
                    vals.append(t["obj"])
            else:
                if t.get("obj_text"):
                    sql += "obj_text=?"
                    vals.append(t["obj_text"])
                else:
                    sql += "obj=?"
                    vals.append(t["obj"])
        sql += " GROUP BY subj HAVING COUNT(subj)=? )"
        vals.append(len(triples))
        return sql, vals

    def get_query(self, search_data, ignore_self=True):
        table = search_data["base_table"]
        if self.is_filter_empty(search_data):
            query = "SELECT uuid FROM " + table
            if ignore_self:
                query += " WHERE uuid !=?"
                return query, [self.self_memid]
            else:
                return query, []

        query = (
            "SELECT {}.uuid FROM {}"
            " INNER JOIN Memories as M on M.uuid={}.uuid"
            " WHERE ".format(table, table, table)
        )

        args = []
        fragment, vals = self.range_queries(search_data.get("base_range", {}), table)
        query = maybe_and(query, len(args) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.exact_matches(search_data.get("base_exact", {}), table)
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.range_queries(search_data.get("memories_range", {}), "M")
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.exact_matches(search_data.get("memories_exact", {}), "M")
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        fragment, vals = self.triples(search_data.get("triples", []), table)
        query = maybe_and(query, len(args) > 0 and len(vals) > 0)
        args.extend(vals)
        query += fragment

        if ignore_self:
            query += " AND {}.uuid !=?".format(table)
            args.append(self.self_memid)
        return query, args

    # flag (default) so that it makes a copy of speaker_look etc so that if the searcher is called
    # later so it doesn't return the new position of the agent/speaker/speakerlook
    # how to parse this distinction?
    def handle_special(self, agent_memory, search_data):
        d = search_data.get("special")
        if not d:
            return []
        if d.get("SPEAKER"):
            return [agent_memory.get_player_by_eid(d["SPEAKER"])]
        if d.get("SPEAKER_LOOK"):
            memids = agent_memory._db_read_one(
                'SELECT uuid FROM ReferenceObjects WHERE ref_type="attention" AND type_name=?',
                d["SPEAKER_LOOK"],
            )
            if memids:
                memid = memids[0]
                mem = agent_memory.get_location_by_id(memid)
                return [mem]
        if d.get("AGENT") is not None:
            return [agent_memory.get_player_by_eid(d["AGENT"])]
        if d.get("DUMMY"):
            return [d["DUMMY"]]
        return []

    def search(self, agent_memory, search_data=None) -> List["MemoryNode"]:  # noqa T484
        """Find ref_objs matching the given filters
        search_data has children:
            "base_table", value is a string with a table name.  if not specified, the
                  base table is ReferenceObjects
            "base_range", dict, with keys "min<column_name>" or "max<column_name>", 
                  (that is the string "min" prepended to the column name)
                  and float values vmin and vmax respectively.  
                  <column_name> is any column in the base table that
                  is a numerical value.  filters on rows satisfying the inequality 
                  <column_entry> > vmin or <column_entry> < vmax
            "base_exact", dict,  with keys "<column_name>"
                  <column_name> is any column in the base table
                  checks exact matches to the value
            "memories_range" and "memories_exact" are the same, but columns in the Memories table
            "triples" list [t0, t1, ...,, tm].  each t in the list is a dict
                  with form t = {"pred_text": <pred>, "obj_text": <obj>}
                  or t = {"pred_text": <pred>, "obj": <obj_memid>}
                  currently returns memories with all triples matched 
        """
        if not search_data:
            search_data = self.search_data
        assert search_data
        search_data["base_table"] = search_data.get("base_table", "ReferenceObjects")
        self.search_data = search_data
        if search_data.get("special"):
            return self.handle_special(agent_memory, search_data)
        query, args = self.get_query(search_data)
        memids = [m[0] for m in agent_memory._db_read(query, *args)]
        return [agent_memory.get_mem_by_id(memid) for memid in memids]


# TODO subclass for filters that return at most one memory,value?
# TODO base_query instead of table
class MemoryFilter:
    def __init__(self, agent_memory, table="ReferenceObjects", preceding=None):
        """
        An object to search an agent memory, or to filter memories.

        args:
            agent_memory: and AgentMemory object
            table (str): a base table for the search, defining the "universe".  
                Should be a table name from the memory schema (and is allowed to be 
                the base memory table)
            preceding (MemoryFilter): if preceding is not None, this MemoryFilter will 
                operate on the output of preceding

        the subclasses of MemoryFilter define three methods, .filter(memids, values)  and .search()
        and a __call__(memids=None, values=None)
        filter returns a subset of the memids and a matching subset of (perhaps transformed) vals
        search takes no input; and instead uses all memories in its table as "input"
        
        The standard interface to the MemoryFilter should be through the __call__
        if the __call__ gets no inputs, its a .search(); otherwise its a .filter on the value and memids
        """
        self.memory = agent_memory
        self.table = table
        self.head = None
        self.is_tail = True
        self.preceding = None
        if preceding:
            self.append(preceding)

    # SO ugly whole object should be a tree or deque, nothing stops previous filters from breaking chain etc.
    # FIXME
    # appends F to right (headwards) of self:
    # self will run on the output of F
    def append(self, F):
        if not self.is_tail:
            raise Exception("don't use MemoryFilter.append when MemoryFilter is not the tail")
        F.is_tail = False
        if not self.head:
            self.preceding = F
        else:
            # head is not guaranteed to be correct except at tail! (middle filters will be wrong)
            self.head.preceding = F
        self.head = F.head or F

    def all_table_memids(self):
        cmd = "SELECT uuid FROM " + self.table
        return [m[0] for m in self.memory._db_read(cmd)]

    # returns a list of memids, a list of vals
    # no input
    # search DOES NOT respect preceding; use the call if you want to respect preceding
    def search(self):
        if not self.preceding:
            all_memids = self.all_table_memids()
            return all_memids, [None] * len(all_memids)
        return [], []

    # inputs a list of memids, a list of vals,
    # returns a list of memids, a list of vals,
    def filter(self, memids, vals):
        return memids, vals

    def __call__(self, mems=None, vals=None):
        if self.preceding:
            mems, vals = self.preceding(mems=mems, vals=vals)
        if mems is None:
            # specifically excluding the case where mems=[]; then should filter
            return self.search()
        else:
            return self.filter(mems, vals)

    def _selfstr(self):
        return self.__class__.__name__

    def __repr__(self):
        if self.preceding:
            return self._selfstr() + " <-- " + str(self.preceding)
        else:
            return self._selfstr()


class NoneTransform(MemoryFilter):
    def __init__(self, agent_memory):
        super().__init__(agent_memory)

    def search(self):
        all_memids = self.all_table_memids()
        return all_memids, [None] * len(all_memids)

    def filter(self, memids, vals):
        return memids, [None] * len(memids)


# FIXME counting blocks in MC will still fail!!!
class CountTransform(MemoryFilter):
    def __init__(self, agent_memory):
        super().__init__(agent_memory)

    def search(self):
        all_memids = self.all_table_memids()
        return all_memids, [len(all_memids)] * len(all_memids)

    def filter(self, memids, vals):
        return memids, [len(memids)] * len(memids)


class ApplyAttribute(MemoryFilter):
    def __init__(self, agent_memory, attribute):
        super().__init__(agent_memory)
        self.attribute = attribute

    def search(self):
        all_memids = self.all_table_memids()
        return all_memids, self.attribute([self.memory.get_mem_by_id(m) for m in all_memids])

    def filter(self, memids, vals):
        return memids, self.attribute([self.memory.get_mem_by_id(m) for m in memids])

    def _selfstr(self):
        return "Apply " + str(self.attribute)


class RandomMemorySelector(MemoryFilter):
    def __init__(self, agent_memory):
        super().__init__(agent_memory)

    def search(self):
        all_memids = self.all_table_memids()
        i = torch.randint(len(all_memids), (1,)).item()
        return [all_memids[i]], [None]

    def filter(self, memids, vals):
        i = torch.randint(len(memids), (1,)).item()
        return [memids[i]], [vals[i]]


class ExtremeValueMemorySelector(MemoryFilter):
    def __init__(self, agent_memory, polarity="argmax", ordinal=1):
        # polarity is "argmax" or "argmin"
        super().__init__(agent_memory)
        self.polarity = polarity

    # should this give an error? probably it should
    def search(self):
        all_memids = self.all_table_memids()
        i = torch.randint(len(all_memids), (1,)).item()
        return [all_memids[i]], [None]

    # FIXME!!!! do ordinals not 1
    def filter(self, memids, vals):
        if not memids:
            return [], []
        if self.polarity == "argmax":
            try:
                i = torch.argmax(torch.Tensor(vals)).item()
            except:
                raise Exception("are these values numbers?  trying to argmax them")
        else:
            try:
                i = torch.argmin(torch.Tensor(vals)).item()
            except:
                raise Exception("are these values numbers?  trying to argmin them")

        return [memids[i]], [vals[i]]

    def _selfstr(self):
        return self.polarity


class LogicalOperationFilter(MemoryFilter):
    def __init__(self, agent_memory, searchers):
        super().__init__(agent_memory)
        self.searchers = searchers
        if not self.searchers:
            raise Exception("empty filter list input into LogicalOperationFilter constructor")


class AndFilter(LogicalOperationFilter):
    def __init__(self, agent_memory, searchers):
        super().__init__(agent_memory, searchers)

    # TODO more efficient?
    # TODO unhashable values
    def search(self):
        N = len(self.searchers)
        mems, vals = self.searchers[0].search()
        for i in range(1, N):
            mems, vals = self.searchers[i].filter(mems, vals)
        return mems, vals

    def filter(self, mems, vals):
        for f in self.searchers:
            mems, vals = f.filter(mems, vals)
        return mems, vals


class OrFilter(LogicalOperationFilter):
    def __init__(self, agent_memory, searchers):
        super().__init__(agent_memory, searchers)

    # TODO more efficient?
    # TODO what to do if same memid has two different values? current version is not commutative
    # TODO warn/error if this has no previous?
    def search(self):
        all_mems = []
        vals = []
        for f in self.searchers:
            mems, v = f.search()
            all_mems.extend(mems)
            vals.extend(v)
        return self.filter(mems, vals)

    def filter(self, memids, vals):
        outmems = {}
        for i in range(len(memids)):
            outmems[memids[i]] = vals[i]
        memids, vals = outmems.items()
        return list(memids), list(vals)


class NotFilter(LogicalOperationFilter):
    def __init__(self, agent_memory, searchers):
        super().__init__(agent_memory, searchers)

    def search(self):
        cmd = "SELECT uuid FROM " + self.base_filters.table
        all_memids = self.memory._db_read(cmd)
        out_memids = list(set(all_memids) - set(self.searchers[0].search()))
        out_vals = [None] * len(out_memids)
        return out_memids, out_vals

    def filter(self, memids, vals):
        mem_dict = dict(zip(memids, vals))
        p_memids, _ = self.searchers[0].filter(memids, vals)
        filtered_memids, _ = list(set(memids) - set(p_memids))
        filtered_vals = [mem_dict[m] for m in filtered_memids]
        return filtered_memids, filtered_vals


class FixedMemFilter(MemoryFilter):
    def __init__(self, agent_memory, memid):
        super().__init__(agent_memory)
        self.memid = memid

    def search(self):
        return [self.memid], [None]

    def filter(self, memids, vals):
        try:
            i = memids.index(self.memid)
            return [memids[i]], vals[i]
        except:
            return [], []


class ComparatorFilter(MemoryFilter):
    def __init__(self, agent_memory, comparator_attribute):
        super().__init__(agent_memory)
        self.comparator = comparator_attribute

    def search(self):
        cmd = "SELECT uuid FROM " + self.base_filters.table
        all_memids = self.memory._db_read(cmd)
        return self.filter(all_memids, [None] * len(all_memids))

    def filter(self, memids, vals):
        mems = [self.memory.get_mem_by_id(m) for m in memids]
        T = self.comparator(mems)
        return (
            [memids[i] for i in range(len(mems)) if T[i]],
            [vals[i] for i in range(len(mems)) if T[i]],
        )


# FIXME!!!! has_x : FILTERS doesn't work
class BasicFilter(MemoryFilter):
    def __init__(self, agent_memory, search_data):
        super().__init__(agent_memory)
        self.search_data = search_data
        self.sql_interface = BasicMemorySearcher(search_data=search_data)

    def get_memids(self):
        return [m.memid for m in self.sql_interface.search(self.memory)]

    def search(self):
        memids = self.get_memids()
        return memids, [None] * len(memids)

    def filter(self, memids, vals):
        acceptable_memids = self.get_memids()
        filtered_memids = [memids[i] for i in range(len(memids)) if memids[i] in acceptable_memids]
        filtered_vals = [vals[i] for i in range(len(memids)) if memids[i] in acceptable_memids]
        return filtered_memids, filtered_vals

    def _selfstr(self):
        return "Basic: (" + str(self.search_data) + ")"


if __name__ == "__main__":
    search_data = {
        "ref_obj_range": {"minx": 3},
        "memories_exact": {"create_time": 1},
        "triples": [
            {"pred_text": "has_tag", "obj_text": "cow"},
            {"pred_text": "has_name", "obj_text": "eddie"},
        ],
    }
