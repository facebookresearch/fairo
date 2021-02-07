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
from location_helpers import interpret_relative_direction
from comparator_helper import interpret_comparator
from dialogue_object_utils import tags_from_dict

CARDINAL_RADIUS = 20


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
        agent_memory = interpreter.agent.memory
        self.get_attribute = interpreter.subinterpret.get("attribute", AttributeInterpreter())
        output = filters_d.get("output")
        val_map = None
        if output and type(output) is dict:
            attr_d = output.get("attribute")
            a = self.get_attribute(interpreter, speaker, attr_d, get_all=get_all)
            val_map = ApplyAttribute(agent_memory, a)
        elif output and output == "count":
            val_map = CountTransform(agent_memory)
        # NB (kavyasrinet) output can be string and have value "memory" too here

        # is this a specific memory?
        # ... then return
        mem, _ = maybe_specific_mem(interpreter, speaker, {"filters": filters_d})
        if mem:
            if val_map:
                val_map.append(FixedMemFilter(agent_memory, mem.memid))
                return val_map
            else:
                return FixedMemFilter(agent_memory, mem.memid)

        F = MemoryFilter(agent_memory)

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
            F.append(BasicFilter(agent_memory, {"triples": triples}))

        selector = None
        location_d = filters_d.get("location")
        if location_d:
            selector = build_linear_extent_selector(interpreter, speaker, location_d)
        else:
            argval_d = filters_d.get("argval")
            if argval_d:
                polarity = "arg" + argval_d.get("polarity").lower()
                attribute_d = argval_d.get("quantity").get("attribute")
                selector_attribute = self.get_attribute(interpreter, speaker, attribute_d)
                # FIXME
                ordinal = {"first": 1, "second": 2, "third": 3}.get(
                    argval_d.get("ordinal", "first").lower(), 1
                )
                sa = ApplyAttribute(agent_memory, selector_attribute)
                selector = ExtremeValueMemorySelector(
                    agent_memory, polarity=polarity, ordinal=ordinal
                )
                selector.append(sa)

        if val_map:
            if selector:
                selector.append(F)
                val_map.append(selector)
            else:
                val_map.append(F)
            return val_map
        else:
            if selector:
                selector.append(F)
                return selector
            return F
