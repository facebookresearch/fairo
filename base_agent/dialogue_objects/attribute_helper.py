"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from copy import deepcopy
from memory_attributes import LinearExtentAttribute, TableColumn, ListAttribute
from memory_values import LinearExtentValue, FixedValue, convert_comparison_value
from base_util import ErrorWithResponse, number_from_span
from base_agent.memory_nodes import ReferenceObjectNode
from dialogue_object_utils import tags_from_dict


"""
Each of these take args:
interpreter:  root interpreter.
speaker (str): The name of the player/human/agent who uttered 
    the chat resulting in this interpreter
d: logical form from semantic parser
"""


def interpret_span_value(interpreter, speaker, d, comparison_measure=None):
    """
    Make a FixedValue object from a number span

    the optional arg comparison_measure should be something that can be handled by
    the convert_comparison_value function
    """
    num = number_from_span(d)
    if num:
        v = FixedValue(interpreter.agent, num)
        # always convert everything to internal units
        # FIXME handle this better
        v = convert_comparison_value(v, comparison_measure)
    else:
        v = FixedValue(interpreter.agent, d)
    return v


def maybe_specific_mem(interpreter, speaker, ref_obj_d):
    """
    check if the reference object logical form corresponds to a ReferenceObject already
    in memory.  used e.g. in Values and Conditions, to distinguish between a ReferenceObject not in 
    memory now but to be searched for when checking the condition
    """
    mem = None
    search_data = None
    if ref_obj_d.get("special_reference"):
        # this is a special ref object, not filters....
        cands = interpreter.subinterpret["reference_objects"](interpreter, speaker, ref_obj_d)
        return cands[0], None
    filters_d = ref_obj_d.get("filters", {})
    coref = filters_d.get("contains_coreference", "NULL")
    if coref != "NULL":
        # this is a particular entity etc, don't search for the mem at check() time
        if isinstance(coref, ReferenceObjectNode):
            mem = coref
        else:
            sub_ref_obj_d = deepcopy(ref_obj_d)
            # del this to stop recursion, otherwise will end up here again
            del sub_ref_obj_d["filters"]["contains_coreference"]
            cands = interpreter.subinterpret["reference_objects"](
                interpreter, speaker, sub_ref_obj_d
            )
        if not cands:
            # FIXME fix this error
            raise ErrorWithResponse("I don't know which objects' attribute you are talking about")
        # TODO if more than one? ask? use the filters?
        else:
            mem = cands[0]
    else:
        # FIXME use FILTERS
        # this object is only defined by the filters and might be different at different moments
        tags = tags_from_dict(filters_d)
        # make a function, reuse code with get_reference_objects FIXME
        search_data = [{"pred_text": "has_tag", "obj_text": tag} for tag in tags]

    return mem, search_data


def interpret_linear_extent(interpreter, speaker, d, force_value=False):
    """
    returns a LinearExtentAttribute or LinearExtentValue.

    the force_value arg forces the output to be a LinearExtentValue
    """
    location_data = {}
    default_frame = getattr(interpreter, "default_frame", "AGENT")
    frame = d.get("frame", default_frame)
    if frame == "SPEAKER":
        frame = speaker
    if type(frame) is dict:
        frame = frame.get("player_span", "unknown_player")
    if frame == "AGENT":
        location_data["frame"] = "AGENT"
    else:
        p = interpreter.agent.memory.get_player_by_name(frame)
        if p:
            location_data["frame"] = p.eid
        else:
            raise ErrorWithResponse("I don't understand in whose frame of reference you mean")
    location_data["relative_direction"] = d.get("relative_direction", "AWAY")
    # FIXME!!!! has_measure

    rd = d.get("source")
    fixed_role = "source"
    if not rd:
        rd = d.get("destination")
        fixed_role = "destination"
    mem, _ = maybe_specific_mem(interpreter, speaker, rd)
    if not mem:
        F = interpreter.subinterpret["filters"](interpreter, speaker, rd)
        location_data["filter"] = F
    L = LinearExtentAttribute(interpreter.agent, location_data, mem=mem, fixed_role=fixed_role)

    # TODO some sort of sanity check here, these should be rare:
    if (d.get("source") and d.get("destination")) or force_value:
        rd = d.get("destination")
        mem = None
        sd = None
        if rd:
            mem, sd = maybe_specific_mem(interpreter, speaker, rd["filters"])
        L = LinearExtentValue(interpreter.agent, L, mem=mem, search_data=sd)

    return L


class AttributeInterpreter:
    def __call__(self, interpreter, speaker, d_attribute, get_all=False):
        if type(d_attribute) is str:
            d_attribute = CANONICALIZE_ATTRIBUTES.get(d_attribute.lower())
            if d_attribute and type(d_attribute) is str:
                return TableColumn(interpreter.agent, d_attribute, get_all=get_all)
            elif d_attribute and type(d_attribute) is list:
                alist = [self.__call__(interpreter, speaker, a) for a in d_attribute]
                if None not in alist:
                    return ListAttribute(interpreter.agent, alist)
        elif d_attribute.get("linear_extent"):
            return interpret_linear_extent(interpreter, speaker, d_attribute["linear_extent"])


CANONICALIZE_ATTRIBUTES = {
    "x": "x",
    "y": "y",
    "z": "z",
    "location": ["x", "y", "z"],
    "ref_type": "ref_type",
    "head_pitch": "pitch",
    "head_yaw": "yaw",  # FIXME!!! prob should have pose type in memory
    "body_yaw": "yaw",  # FIXME!!! prob should have pose type in memory
    "name": "has_name",
    "has_name": "has_name",
    "has_colour": "has_colour",
    "has_tag": "has_tag",
    "born_time": "create_time",
    "modify_time": "updated_time",
    "visit_time": "attended_time",  # FIXME!!
    "action_name": "action_name",
    # "speaker","finished_time", chat, logical_form ... tasks not supported yet
}
