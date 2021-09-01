"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from copy import deepcopy
from .attribute_helper import AttributeInterpreter, maybe_specific_mem
from droidlet.memory.memory_attributes import LinearExtentAttribute
from droidlet.memory.memory_filters import (
    MemoryFilter,
    AndFilter,
    OrFilter,
    NotFilter,
    ComparatorFilter,
    MemidList,
    FixedMemFilter,
    BasicFilter,
    ApplyAttribute,
    CountTransform,
    ExtremeValueMemorySelector,
    RandomMemorySelector,
    BackoffFilter,
)
from droidlet.base_util import number_from_span
from droidlet.shared_data_structs import ErrorWithResponse
from .location_helpers import interpret_relative_direction
from .comparator_helper import interpret_comparator
from .interpreter_utils import backoff_where

CARDINAL_RADIUS = 20

####################################################################################
### FIXME!!! this file, comparator_helper, etc  needs to be refactored to make use of
### updated memory searcher class
### lots of overlapping code
####################################################################################


def get_val_map(interpreter, speaker, filters_d, get_all=False):
    output = filters_d.get("output")
    val_map = None
    if output and type(output) is dict:
        attr_d = output.get("attribute")
        get_attribute = interpreter.subinterpret.get("attribute", AttributeInterpreter())
        a = get_attribute(interpreter, speaker, attr_d, get_all=get_all)
        val_map = ApplyAttribute(interpreter.memory, a)
    elif output and output == "COUNT":
        val_map = CountTransform(interpreter.memory)
    return val_map


def maybe_append_left(F, to_append=None):
    if to_append is not None:
        to_append.append(F)
        return to_append
    else:
        return F


def maybe_handle_specific_mem(interpreter, speaker, filters_d, val_map):
    # is this a specific memory?
    # ... then return it
    F = None
    if filters_d.get("special") and filters_d["special"] == "THIS":
        F = FixedMemFilter(interpreter.memory, "NULL")
    else:
        mem, _ = maybe_specific_mem(interpreter, speaker, {"filters": filters_d})
        if mem:
            F = FixedMemFilter(interpreter.memory, mem.memid)
    if F is not None:
        return maybe_append_left(F, to_append=val_map)
    else:
        return None


def interpret_where_clause(
    interpreter, speaker, where_d, memory_type="ReferenceObject", ignore_self=False
):
    """ 
    where_d is a sentence (nested dict/list) of the recursive form 
    COMPARATOR, TRIPLE, or {CONJUNCTION, [where_clauses]}
    where each CONJUCTION is either "AND", "OR", or "NOT"
    """
    subclause = where_d.get("AND") or where_d.get("OR") or where_d.get("NOT")
    if subclause:
        clause_filters = []
        for c in subclause:
            clause_filters.append(
                interpret_where_clause(
                    interpreter, speaker, c, memory_type=memory_type, ignore_self=ignore_self
                )
            )
        if "AND" in where_d:
            return AndFilter(interpreter.memory, clause_filters)
        elif "OR" in where_d:
            return OrFilter(interpreter.memory, clause_filters)
        else:  # NOT
            try:
                assert len(clause_filters) == 1
            except:
                raise Exception(
                    "tried to make a NOT filter with a list of clauses longer than 1 {}".format(
                        where_d
                    )
                )
            return NotFilter(interpreter.memory, clause_filters)
    if where_d.get("input_left"):
        # this is a comparator leaf
        comparator_attribute = interpret_comparator(
            interpreter, speaker, where_d, is_condition=False
        )
        return ComparatorFilter(interpreter.memory, comparator_attribute, memtype=memory_type)
    else:
        # this is a triple leaf
        query = {"memory_type": memory_type, "where_clause": {"AND": [{}]}}
        for k, v in where_d.items():
            if type(v) is dict:
                query["where_clause"]["AND"][0][k] = interpreter.subinterpet["filters"](
                    interpreter, speaker, v
                )
            else:
                query["where_clause"]["AND"][0][k] = v
        return BasicFilter(interpreter.memory, query, ignore_self=ignore_self)


def interpret_where_backoff(
    interpreter, speaker, where_d, memory_type="ReferenceObject", ignore_self=False
):
    F = interpret_where_clause(
        interpreter, speaker, where_d, memory_type=memory_type, ignore_self=ignore_self
    )
    _, modified_where = backoff_where(where_d)
    G = interpret_where_clause(
        interpreter, speaker, modified_where, memory_type=memory_type, ignore_self=ignore_self
    )
    return BackoffFilter(interpreter.memory, [F, G])


def interpret_random_selector(interpreter, speaker, selector_d):
    """
    returns a RandomMemorySelector from the selector_d
    """
    random_num = selector_d.get("ordinal", 1)
    n = number_from_span(random_num)
    try:
        n = int(n)
    except:
        raise Exception(
            "malformed selector dict {}, tried to get random number {} ".format(
                selector_d, random_num
            )
        )
    s = selector_d.get("same", "ALLOWED")
    return RandomMemorySelector(interpreter.memory, same=s, n=n)


def interpret_argval_selector(interpreter, speaker, selector_d):
    return_d = selector_d.get("return_quantity", {})
    argval_d = return_d.get("argval")
    if not argval_d:
        raise Exception(
            "tried to build argval selector from logical form without argval clause {}".format(
                selector_d
            )
        )
    polarity = "arg" + argval_d.get("polarity").lower()
    attribute_d = argval_d.get("quantity").get("attribute")
    get_attribute = interpreter.subinterpret.get("attribute", AttributeInterpreter())
    selector_attribute = get_attribute(interpreter, speaker, attribute_d)
    # FIXME
    ordinal = {"first": 1, "second": 2, "third": 3}.get(
        argval_d.get("ordinal", "first").lower(), 1
    )
    sa = ApplyAttribute(interpreter.memory, selector_attribute)
    selector = ExtremeValueMemorySelector(interpreter.memory, polarity=polarity, ordinal=ordinal)
    selector.append(sa)
    return selector


def build_linear_extent_selector(interpreter, speaker, location_d):
    """
    builds a MemoryFilter that selects by a linear_extent dict
    chooses memory location nearest to
    the linear_extent dict interpreted as a location
    """

    # FIXME this is being done at construction time, rather than execution
    mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location_d)
    steps, reldir = interpret_relative_direction(interpreter, location_d)
    pos, _ = interpreter.subinterpret["specify_locations"](
        interpreter, speaker, mems, steps, reldir
    )

    class dummy_loc_mem:
        def get_pos(self):
            return pos

    selector_attribute = LinearExtentAttribute(
        interpreter.memory, {"relative_direction": "AWAY"}, mem=dummy_loc_mem()
    )
    polarity = "argmin"
    sa = ApplyAttribute(interpreter.memory, selector_attribute)
    selector = ExtremeValueMemorySelector(interpreter.memory, polarity=polarity, ordinal=1)
    selector.append(sa)
    mems_filter = MemidList(interpreter.memory, [mems[0].memid])
    not_mems_filter = NotFilter(interpreter.memory, [mems_filter])
    selector.append(not_mems_filter)
    #    selector.append(build_radius_comparator(interpreter, speaker, location_d))

    return selector


def interpret_selector(interpreter, speaker, selector_d):
    selector = None
    if selector_d.get("location"):
        return build_linear_extent_selector(interpreter, speaker, selector_d["location"])
    return_d = selector_d.get("return_quantity", "ALL")
    if type(return_d) is str:
        if return_d == "ALL":
            # no selector, just return everything
            pass
        elif return_d == "RANDOM":
            selector = interpret_random_selector(interpreter, speaker, selector_d)
        else:
            raise Exception("malformed selector dict {}".format(selector_d))
    else:
        argval_d = return_d.get("argval")
        if argval_d:
            selector = interpret_argval_selector(interpreter, speaker, selector_d)
        else:
            raise Exception("malformed selector dict {}".format(selector_d))
    return selector


def maybe_apply_selector(interpreter, speaker, filters_d, F):
    selector_d = filters_d.get("selector", {})
    selector = interpret_selector(interpreter, speaker, selector_d)
    if selector is not None:
        selector.append(F)
        return selector
    else:
        return F


# FIXME!  update DSL so this is unnecessary
def convert_task_where(where_clause):
    """
    converts special tags for tasks to comparators.
    REMOVE ME ASAP!!
    returns the modified where_clause dict
    """
    new_where_clause = deepcopy(where_clause)
    # doesn't check if where_clause is well formed
    subwhere = where_clause.get("AND") or where_clause.get("OR") or where_clause.get("NOT")
    if subwhere:  # recurse
        conj = list(where_clause.keys())[0]
        for i in range(len(subwhere)):
            new_where_clause[conj][i] = convert_task_where(where_clause[conj][i])
    else:  # a leaf
        if "input_left" in where_clause:  # a comparator, leave alone
            pass
        else:  # triple...
            o = where_clause.get("obj_text").lower()
            if o == "currently_running":
                new_where_clause = {
                    "input_left": {"value_extractor": {"attribute": "running"}},
                    "input_right": {"value_extractor": "1"},
                    "comparison_type": "EQUAL",
                }
            elif o == "paused":
                new_where_clause = {
                    "input_left": {"value_extractor": {"attribute": "paused"}},
                    "input_right": {"value_extractor": "1"},
                    "comparison_type": "EQUAL",
                }
            elif o == "finished":
                new_where_clause = {
                    "input_left": {"value_extractor": {"attribute": "finished"}},
                    "input_right": {"value_extractor": "0"},
                    "comparison_type": "GREATER_THAN",
                }
            else:
                new_where_clause["obj_text"] = o
    return new_where_clause


def interpret_task_filter(interpreter, speaker, filters_d, get_all=False):
    modified_where = convert_task_where(filters_d.get("where_clause", {}))
    return interpret_where_clause(interpreter, speaker, modified_where, memory_type="Task")


def interpret_dance_filter(interpreter, speaker, filters_d, get_all=False):
    return interpret_where_backoff(
        interpreter, speaker, filters_d.get("where_clause", {}), memory_type="Dance"
    )


class FilterInterpreter:
    def __call__(self, interpreter, speaker, filters_d, get_all=False):
        """
        This is a subinterpreter to handle FILTERS dictionaries

        Args:
        interpreter:  root interpreter.
        speaker (str): The name of the player/human/agent who uttered
            the chat resulting in this interpreter
        filters_d: FILTERS logical form from semantic parser
        get_all (bool): if True, output attributes are set with get_all=True

        Outputs a (chain) of MemoryFilter objects
        """
        val_map = get_val_map(interpreter, speaker, filters_d, get_all=get_all)
        # NB (kavyasrinet) output can be string and have value "memory" too here

        # is this a specific memory?
        # ... then return
        specific_mem_filter = maybe_handle_specific_mem(interpreter, speaker, filters_d, val_map)
        if specific_mem_filter is not None:
            return specific_mem_filter
        memtype = filters_d.get("memory_type", "REFERENCE_OBJECT")
        # FIXME/TODO: these share lots of code, refactor
        if memtype == "REFERENCE_OBJECT":
            # just using this to check if SELF is a possibility, TODO finer control
            tags, _ = backoff_where(filters_d.get("where_clause"), {})
            F = interpret_where_backoff(
                interpreter,
                speaker,
                filters_d.get("where_clause", {}),
                memory_type="ReferenceObject",
                ignore_self=not ("SELF" in tags),
            )
        elif memtype == "TASKS":
            F = interpret_task_filter(interpreter, speaker, filters_d)
        else:
            memtype_key = memtype.lower() + "_filters"
            try:
                F = interpreter.subinterpret[memtype_key](interpreter, speaker, filters_d)
            except:
                raise ErrorWithResponse(
                    "failed at interpreting filters of type {}".format(memtype)
                )
        F = maybe_apply_selector(interpreter, speaker, filters_d, F)
        return maybe_append_left(F, to_append=val_map)
