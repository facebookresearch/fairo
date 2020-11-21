"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from attribute_helper import AttributeInterpreter, maybe_specific_mem
from memory_filters import (
    MemoryFilter,
    FixedMemFilter,
    BasicFilter,
    ApplyAttribute,
    CountTransform,
    ExtremeValueMemorySelector,
)
from comparator_helper import interpret_comparator
from dialogue_object_utils import tags_from_dict


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
            a = self.get_attribute(interpreter, speaker, output.get("attribute"), get_all=get_all)
            if a:
                val_map = ApplyAttribute(agent_memory, a)
        elif output and output == "count":
            val_map = CountTransform(agent_memory)

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

        if triples:
            F.append(BasicFilter(agent_memory, {"triples": triples}))

        location_d = filters_d.get("location")
        linear_extent_d = {}
        if location_d:
            if location_d.get("relative_direction"):
                linear_extent_d["relative_direction"] = location_d.get("relative_direction")
            linear_extent_d["frame"] = "SPEAKER"
            linear_extent_d["source"] = location_d["reference_object"]
            # FIXME deal with coref resolve in location if its a mem

        selector_attribute = None
        if linear_extent_d:
            selector_attribute = self.get_attribute(
                interpreter, speaker, {"linear_extent": linear_extent_d}
            )
            polarity = "argmin"  # fIXME for inside?
            ordinal = 1

        argval_d = filters_d.get("argmin") or filters_d.get("argmax")
        if argval_d:
            polarity = "argmin" if "argmin" in filters_d.keys() else "argmax"
            attribute_d = argval_d["quantity"].get("attribute", "NULL")
            selector_attribute = self.get_attribute(interpreter, speaker, attribute_d)
            # FIXME
            ordinal = {"first": 1, "second": 2, "third": 3}.get(
                argval_d.get("ordinal", "first").lower(), 1
            )

        selector = None
        if selector_attribute:
            sa = ApplyAttribute(agent_memory, selector_attribute)
            selector = ExtremeValueMemorySelector(agent_memory, polarity=polarity, ordinal=ordinal)
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
