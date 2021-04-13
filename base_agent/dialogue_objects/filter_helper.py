"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from attribute_helper import AttributeInterpreter, maybe_specific_mem
from memory_attributes import LinearExtentAttribute
from memory_filters import (
    MemoryFilter,
    NotFilter,
    MemidList,
    FixedMemFilter,
    BasicFilter,
    ApplyAttribute,
    CountTransform,
    ExtremeValueMemorySelector,
)
from base_agent.base_util import ErrorWithResponse
from location_helpers import interpret_relative_direction
from comparator_helper import interpret_comparator
from dialogue_object_utils import tags_from_dict

CARDINAL_RADIUS = 20


def get_val_map(interpreter, speaker, filters_d, get_all=False):
    output = filters_d.get("output")
    val_map = None
    if output and type(output) is dict:
        attr_d = output.get("attribute")
        get_attribute = interpreter.subinterpret.get("attribute", AttributeInterpreter())
        a = get_attribute(interpreter, speaker, attr_d, get_all=get_all)
        val_map = ApplyAttribute(interpreter.agent.memory, a)
    elif output and output == "COUNT":
        val_map = CountTransform(interpreter.agent.memory)
    return val_map


def maybe_append_left(F, to_append=None):
    if to_append is not None:
        to_append.append(F)
        return to_append
    else:
        return F


def maybe_handle_specific_mem(interpreter, speaker, filters_d, val_map):
    # is this a specific memory?
    # ... then return
    mem, _ = maybe_specific_mem(interpreter, speaker, {"filters": filters_d})
    if mem:
        return maybe_append_left(
            FixedMemFilter(interpreter.agent.memory, mem.memid), to_append=val_map
        )
    else:
        return None


def interpret_ref_obj_filter(interpreter, speaker, filters_d):
    F = MemoryFilter(interpreter.agent.memory)

    # currently spec intersects all comparators TODO?
    comparator_specs = filters_d.get("comparator")
    if comparator_specs:
        for s in comparator_specs:
            F.append(interpret_comparator(interpreter, speaker, s, is_condition=False))

    # FIXME!!! AUTHOR
    # FIXME!!! has_x=FILTERS
    # currently spec intersects all has_x, TODO?
    # FIXME!!! tags_from_dict is crude, use tags/relations appropriately
    #        triples = []
    tags = tags_from_dict(filters_d)
    triples = [{"pred_text": "has_tag", "obj_text": tag} for tag in tags]
    #        for k, v in filters_d.items():
    #            if type(k) is str and "has" in k:
    #                if type(v) is str:
    #                    triples.append({"pred_text": k, "obj_text": v})
    # Warning: BasicFilters will filter out agent's self
    # FIXME !! finer control over this ^
    if triples:
        F.append(BasicFilter(interpreter.agent.memory, {"triples": triples}))

    return F


def interpret_random_selector(interpreter, speaker, selector_d):
    """
    returns a RandomMemorySelector from the selector_d
    """
    return_d = selector_d.get("return_quantity", {})
    random_num = return_d.get("random")
    if not random_num:
        raise Exception(
            "tried to build random selector from logical form without random clause {}".format(
                selector_d
            )
        )
    try:
        if type(random_num) is int:
            n = random_num
        else:
            n = word_to_num(random_num)
    except:
        raise Exception(
            "malformed selector dict {}, tried to get random number {} ".format(
                selector_d, random_num
            )
        )
    s = selector_d.get("same", "ALLOWED")
    return RandomMemorySelector(interpreter.agent.memory, same=s, n=n)


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
    sa = ApplyAttribute(interpreter.agent.memory, selector_attribute)
    selector = ExtremeValueMemorySelector(
        interpreter.agent.memory, polarity=polarity, ordinal=ordinal
    )
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
        interpreter.agent, {"relative_direction": "AWAY"}, mem=dummy_loc_mem()
    )
    polarity = "argmin"
    sa = ApplyAttribute(interpreter.agent.memory, selector_attribute)
    selector = ExtremeValueMemorySelector(interpreter.agent.memory, polarity=polarity, ordinal=1)
    selector.append(sa)
    mems_filter = MemidList(interpreter.agent.memory, [mems[0].memid])
    not_mems_filter = NotFilter(interpreter.agent.memory, [mems_filter])
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
        else:
            raise Exception("malformed selector dict {}".format(selector_d))
    else:
        argval_d = return_d.get("argval")
        random_num = return_d.get("random")
        if argval_d:
            selector = interpret_argval_selector(interpreter, speaker, selector_d)
        elif random_num:
            selector = interpret_random_selector(interpreter, speaker, selector_d)
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


def interpret_task_filter(interpreter, speaker, filters_d, get_all=False):
    F = MemoryFilter(interpreter.agent.memory)

    task_tags = ["currently_running", "running", "paused", "finished"]

    T = filters_d.get("triples")
    task_properties = [
        a.get("obj_text").lower() for a in T if a.get("obj_text", "").lower() in task_tags
    ]
    search_data = {}
    search_data["base_table"] = "Tasks"
    search_data["base_exact"] = {}
    if "currently_running" in task_properties:
        search_data["base_exact"]["running"] = 1
    if "paused" in task_properties:
        search_data["base_exact"]["paused"] = 1
    else:
        search_data["base_exact"]["paused"] = 0
    search_data["base_range"] = {}
    if "finished" in task_properties:
        search_data["base_range"]["minfinished"] = 0
    F.append(BasicFilter(interpreter.agent.memory, search_data))

    # currently spec intersects all comparators TODO?
    comparator_specs = filters_d.get("comparator")
    if comparator_specs:
        for s in comparator_specs:
            F.append(interpret_comparator(interpreter, speaker, s, is_condition=False))

    return F


def interpret_dance_filter(interpreter, speaker, filters_d, get_all=False):
    F = MemoryFilter(interpreter.agent.memory)
    search_data = {}
    triples = []
    for t in filters_d.get("triples"):
        triples.append(t)
    search_data["base_table"] = "Dances"
    search_data["triples"] = triples
    F.append(BasicFilter(interpreter.agent.memory, search_data))
    # currently spec intersects all comparators TODO?
    comparator_specs = filters_d.get("comparator")
    if comparator_specs:
        for s in comparator_specs:
            F.append(interpret_comparator(interpreter, speaker, s, is_condition=False))
    return F


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
        specific_mem = maybe_handle_specific_mem(interpreter, speaker, filters_d, val_map)
        if specific_mem is not None:
            return specific_mem

        memtype = filters_d.get("memory_type", "REFERENCE_OBJECT")
        # FIXME/TODO: these share lots of code, refactor
        if memtype == "REFERENCE_OBJECT":
            F = interpret_ref_obj_filter(interpreter, speaker, filters_d)
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
