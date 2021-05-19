"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import re
import numpy as np
from copy import deepcopy
from typing import cast, List, Tuple, Dict


from .dialogue_object_utils import SPEAKERLOOK
from .dialogue_object import ConfirmReferenceObject
from .location_helpers import interpret_relative_direction
from droidlet.base_util import euclid_dist, number_from_span
from droidlet.memory.memory_attributes import LookRayDistance, LinearExtentAttribute
from droidlet.memory.memory_nodes import ReferenceObjectNode
from droidlet.base_util import T, XYZ
from ...shared_data_structs import ErrorWithResponse, NextDialogueStep
from droidlet.dialog.dialogue_objects.filter_helper import interpret_selector


def get_eid_from_special(agent_memory, S="AGENT", speaker=None):
    """get the entity id corresponding to a special ReferenceObject"""
    eid = None
    if S == "SPEAKER_LOOK" or S == "SPEAKER":
        if not speaker:
            raise Exception("Asked for speakers memid but did not give speaker name")
        eid = agent_memory.get_player_by_name(speaker).eid
    elif S == "AGENT":
        eid = agent_memory.get_mem_by_id(agent_memory.self_memid).eid
    return eid


def special_reference_search_data(interpreter, speaker, S, entity_id=None, agent_memory=None):
    """make a search dictionary for a BasicMemorySearcher to return the special ReferenceObject"""
    # TODO/FIXME! add things to workspace memory
    agent_memory = agent_memory or interpreter.agent.memory
    if type(S) is dict:
        coord_span = S["coordinates_span"]
        loc = cast(XYZ, tuple(int(float(w)) for w in re.findall("[-0-9.]+", coord_span)))
        if len(loc) != 3:
            logging.error("Bad coordinates: {}".format(coord_span))
            raise ErrorWithResponse("I don't understand what location you're referring to")
        memid = agent_memory.add_location((int(loc[0]), int(loc[1]), int(loc[2])))
        mem = agent_memory.get_location_by_id(memid)
        f = {"special": {"DUMMY": mem}}
    else:
        f = {"special": {S: entity_id}}
    return f


def get_special_reference_object(interpreter, speaker, S, agent_memory=None, eid=None):
    """subinterpret a special reference object.
    args:
    interpreter:  the root interpreter
    speaker (str): The name of the player/human/agent who uttered
        the chat resulting in this interpreter
    S:  the special reference object logical form from
    """

    # TODO/FIXME! add things to workspace memory
    agent_memory = agent_memory or interpreter.agent.memory
    if not eid:
        eid = get_eid_from_special(agent_memory, S, speaker=speaker)
    sd = special_reference_search_data(None, speaker, S, entity_id=eid, agent_memory=agent_memory)
    mems = agent_memory.basic_search(sd)
    if not mems:
        # need a better interface for this, don't need to run full perception
        # just to force speakerlook in memory
        # TODO force if look is stale, not just if it doesn't exist
        # this branch shouldn't occur
        # interpreter.agent.perceive(force=True)
        raise ErrorWithResponse(
            "I think you are pointing at something but I don't know what it is"
        )
    return mems[0]


###########################################################################
# FIXME!!!!! rewrite interpret_reference_object, filter_by_sublocation,
#            ReferenceLocationInterpreter to use FILTERS cleanly
#            current system is ungainly and wrong...
#            interpretation of selector and filtering by location
#            is spread over the above objects and functions in filter_helper
###########################################################################
class ReferenceObjectInterpreter:
    def __init__(self, interpret_reference_object):
        self.interpret_reference_object = interpret_reference_object

    def __call__(self, *args, **kwargs):
        return self.interpret_reference_object(*args, **kwargs)


def interpret_reference_object(
    interpreter,
    speaker,
    d,
    extra_tags=[],
    loose_speakerlook=False,
    allow_clarification=True,
    all_proximity=100,
) -> List[ReferenceObjectNode]:
    """this tries to find a ref obj memory matching the criteria from the
    ref_obj_dict

    args:
    interpreter:  root interpreter.
    speaker (str): The name of the player/human/agent who uttered
        the chat resulting in this interpreter
    d: logical form from semantic parser

    extra_tags (list of strings): tags added by parent to narrow the search
    allow_clarification (bool): should a Clarification object be put on the DialogueStack
    """
    filters_d = d.get("filters")
    special = d.get("special_reference")
    # filters_d can be empty...
    assert (
        filters_d is not None
    ) or special, "no filters or special_reference sub-dicts {}".format(d)
    if special:
        mem = get_special_reference_object(interpreter, speaker, special)
        return [mem]

    if filters_d.get("contains_coreference", "NULL") != "NULL":
        mem = filters_d["contains_coreference"]
        if isinstance(mem, ReferenceObjectNode):
            return [mem]
        elif mem == "resolved":
            pass
        else:
            logging.error("bad coref_resolve -> {}".format(mem))

    if len(interpreter.progeny_data) == 0:
        if any(extra_tags):
            if not filters_d.get("triples"):
                filters_d["triples"] = []
            for tag in extra_tags:
                filters_d["triples"].append({"pred_text": "has_tag", "obj_text": tag})
        # TODO Add ignore_player maybe?

        # FIXME! see above.  currently removing selector to get candidates, and filtering after
        # instead of letting filter interpreters handle.
        filters_no_select = deepcopy(filters_d)
        filters_no_select.pop("selector", None)
        #        filters_no_select.pop("location", None)
        candidate_mems = apply_memory_filters(interpreter, speaker, filters_no_select)
        if len(candidate_mems) > 0:
            return filter_by_sublocation(
                interpreter,
                speaker,
                candidate_mems,
                d,
                loose=loose_speakerlook,
                all_proximity=all_proximity,
            )

        elif allow_clarification:
            # no candidates found; ask Clarification
            # TODO: move ttad call to dialogue manager and remove this logic
            interpreter.action_dict_frozen = True
            confirm_candidates = apply_memory_filters(interpreter, speaker, filters_d)
            objects = object_looked_at(interpreter.agent, confirm_candidates, speaker=speaker)
            if len(objects) == 0:
                raise ErrorWithResponse("I don't know what you're referring to")
            _, mem = objects[0]
            interpreter.provisional["object_mem"] = mem
            interpreter.provisional["filters_d"] = filters_d
            interpreter.dialogue_stack.append_new(ConfirmReferenceObject, mem)
            raise NextDialogueStep()
        else:
            raise ErrorWithResponse("I don't know what you're referring to")

    else:
        # clarification answered
        r = interpreter.progeny_data[-1].get("response")
        if r == "yes":
            # TODO: learn from the tag!  put it in memory!
            return [interpreter.provisional.get("object_mem")]
        else:
            raise ErrorWithResponse("I don't know what you're referring to")


def apply_memory_filters(interpreter, speaker, filters_d) -> List[ReferenceObjectNode]:
    """Return a list of (xyz, memory) tuples encompassing all possible reference objects"""
    F = interpreter.subinterpret["filters"](interpreter, speaker, filters_d)
    memids, _ = F()
    mems = [interpreter.agent.memory.get_mem_by_id(i) for i in memids]
    #    f = {"triples": [{"pred_text": "has_tag", "obj_text": tag} for tag in tags]}
    #    mems = interpreter.memory.basic_search(f)
    return mems


# FIXME make me a proper filters object
# TODO filter by INSIDE/AWAY/NEAR
def filter_by_sublocation(
    interpreter, speaker, candidates: List[T], d: Dict, all_proximity=10, loose=False
) -> List[T]:
    """Select from a list of candidate reference_object mems given a sublocation
    also handles random sampling
    Returns a list of mems
    """
    filters_d = d.get("filters")
    assert filters_d is not None, "no filters: {}".format(d)
    default_loc = getattr(interpreter, "default_loc", SPEAKERLOOK)

    # FIXME! remove this when DSL gets rid of location as selector:
    get_nearest = False
    if filters_d.get("location") and not filters_d["location"].get("relative_direction"):
        get_nearest = True

    location = filters_d.get("location", default_loc)
    reldir = location.get("relative_direction")
    distance_sorted = False
    location_filtered_candidates = []
    if reldir:
        if reldir == "INSIDE":
            # FIXME formalize this better, make extensible
            if location.get("reference_object"):
                # should probably return from interpret_reference_location...
                ref_mems = interpret_reference_object(
                    interpreter, speaker, location["reference_object"]
                )
                I = interpreter.agent.on_demand_perception.get["check_inside"]
                if I:
                    for candidate_mem in candidates:
                        if I([candidate_mem, ref_mems[0]]):
                            location_filtered_candidates.append(candidate_mem)
                else:
                    raise ErrorWithResponse("I don't know how to check inside")
            if not location_filtered_candidates:
                raise ErrorWithResponse("I can't find something inside that")
        elif reldir == "AWAY":
            raise ErrorWithResponse("I don't know which object you mean")
        elif reldir == "NEAR":
            pass  # fall back to no reference direction
        elif reldir == "BETWEEN":
            mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location)
            steps, reldir = interpret_relative_direction(interpreter, d)
            ref_loc, _ = interpreter.subinterpret["specify_locations"](
                interpreter, speaker, mems, steps, reldir
            )
            distance_sorted = True
            location_filtered_candidates = candidates
            location_filtered_candidates.sort(key=lambda c: euclid_dist(c.get_pos(), ref_loc))

        else:
            # reference object location, i.e. the "X" in "left of X"
            mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location)
            if not mems:
                raise ErrorWithResponse("I don't know which object you mean")

            # FIXME!!! handle frame better, might want agent's frame instead
            # FIXME use the subinterpreter, don't directly call the attribute
            eid = interpreter.agent.memory.get_player_by_name(speaker).eid
            self_mem = interpreter.agent.memory.get_mem_by_id(interpreter.agent.memory.self_memid)
            L = LinearExtentAttribute(
                interpreter.agent, {"frame": eid, "relative_direction": reldir}, mem=self_mem
            )
            c_proj = L(candidates)
            m_proj = L(mems)
            # FIXME don't just take the first...
            m_proj = m_proj[0]

            # filter by relative dir, e.g. "left of Y"
            location_filtered_candidates = [c for (p, c) in zip(c_proj, candidates) if p > m_proj]
            # "the X left of Y" = the right-most X that is left of Y
            location_filtered_candidates.sort(key=lambda p: p.get_pos())
            distance_sorted = True
    else:
        # no reference direction: choose the closest
        mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location)
        steps, reldir = interpret_relative_direction(interpreter, d)
        ref_loc, _ = interpreter.subinterpret["specify_locations"](
            interpreter, speaker, mems, steps, reldir
        )
        location_filtered_candidates = [
            c for c in candidates if euclid_dist(c.get_pos(), ref_loc) <= all_proximity
        ]
        location_filtered_candidates.sort(key=lambda c: euclid_dist(c.get_pos(), ref_loc))
        distance_sorted = True
        # FIXME! remove this when DSL gets rid of location as selector:
        if get_nearest:  # this happens if a location was given in input, but no reldir
            location_filtered_candidates = location_filtered_candidates[:1]

    # FIXME lots of copied code between here and interpret_selector
    mems = location_filtered_candidates
    if location_filtered_candidates:  # could be [], if so will return []
        selector_d = filters_d.get("selector", {})
        return_d = selector_d.get("return_quantity", "ALL")
        if type(return_d) is dict and return_d.get("random") and distance_sorted:
            # FIXME? not going to follow spec here, grabbing by the sort...
            try:
                return_num = int(number_from_span(return_d["random"]))
            except:
                raise Exception(
                    "malformed selector dict {}, tried to get number from return dict".format(
                        return_d
                    )
                )
            same = selector_d.get("same", "ALLOWED")
            if same == "REQUIRED":
                mems = [location_filtered_candidates[0]] * return_num
            elif same == "DISALLOWED":
                if return_num > len(location_filtered_candidates):
                    raise ErrorWithResponse(
                        "tried to get {} objects when I only can think of {}".format(
                            return_num, len(location_filtered_candidates)
                        )
                    )
                else:
                    mems = location_filtered_candidates[:return_num]
            else:
                repeat_num = max(return_num - len(location_filtered_candidates), 0)
                return_num = min(return_num, len(location_filtered_candidates))
                mems = (
                    location_filtered_candidates[:return_num]
                    + location_filtered_candidates[0] * repeat_num
                )
        else:
            S = interpret_selector(interpreter, speaker, selector_d)
            if S:
                memids, _ = S(
                    [c.memid for c in location_filtered_candidates],
                    [None] * len(location_filtered_candidates),
                )
                mems = [interpreter.agent.memory.get_mem_by_id(m) for m in memids]
    return mems


def object_looked_at(
    agent, candidates: List[T], speaker=None, eid=None, limit=1, max_distance=30, loose=False
) -> List[Tuple[XYZ, T]]:
    """Return the object that `player` is looking at

    Args:
    - agent
    - candidates: list of memory objects
    - eid or speaker, who is doing the looking
    - limit: 'ALL' or int; max candidates to return
    - loose:  if True, don't filter candaidates behind agent

    Returns: a list of (xyz, mem) tuples, max length `limit`
    """
    if len(candidates) == 0:
        return []
    assert eid or speaker
    if not eid:
        eid = agent.memory.get_player_by_name(speaker).eid
    # TODO wrap in try/catch, handle failures in finding speaker or not having speakers LOS
    xsect = capped_line_of_sight(agent, eid=eid, cap=25)
    speaker_mem = agent.memory.basic_search({"special": {"SPEAKER": eid}})[0]
    pos = np.array(speaker_mem.get_pos())
    yaw, pitch = speaker_mem.get_yaw_pitch()

    def coord(mem):
        return agent.coordinate_transforms.transform(np.array(mem.get_pos()) - pos, yaw, pitch)

    FRONT = agent.coordinate_transforms.DIRECTIONS["FRONT"]
    LEFT = agent.coordinate_transforms.DIRECTIONS["LEFT"]
    UP = agent.coordinate_transforms.DIRECTIONS["UP"]

    # reject objects behind player or not in cone of sight (but always include
    # an object if it's directly looked at)
    if not loose:
        candidates_ = [
            c
            for c in candidates
            if tuple(xsect) in getattr(c, "blocks", {})  # FIXME lopri rename
            or coord(c) @ FRONT > ((coord(c) @ LEFT) ** 2 + (coord(c) @ UP) ** 2) ** 0.5
        ]
    else:
        candidates_ = candidates

    # if looking directly at an object, sort by proximity to look intersection
    if np.linalg.norm(pos - xsect) <= 25:
        candidates_.sort(key=lambda c: np.linalg.norm(np.array(c.get_pos()) - xsect))
    else:
        # otherwise, sort by closest to look vector
        raydists = list(zip(candidates_, LookRayDistance(agent, eid)(candidates_)))
        raydists.sort(key=lambda x: x[1])
        candidates_ = [c[0] for c in raydists]
    # limit returns of things too far away
    candidates_ = [c for c in candidates_ if np.linalg.norm(pos - c.get_pos()) < max_distance]
    # limit number of returns
    if limit == "ALL":
        limit = len(candidates_)
    # FIXME do we need to return postions here? go through code and fix
    return [(c.get_pos(), c) for c in candidates_[:limit]]


def capped_line_of_sight(agent, speaker=None, eid=None, cap=20):
    """Return the location directly in the entity's line of sight, or a point in the distance
    if LOS does not intersect nearby point"""

    assert eid or speaker
    if not eid:
        eid = agent.memory.get_player_by_name(speaker).eid

    xsect_mem = get_special_reference_object(
        None, speaker, "SPEAKER_LOOK", agent_memory=agent.memory, eid=eid
    )
    speaker_mem = agent.memory.basic_search({"special": {"SPEAKER": eid}})[0]
    pos = speaker_mem.get_pos()
    if xsect_mem and np.linalg.norm(np.subtract(xsect_mem.get_pos(), pos)) <= cap:
        return np.array(xsect_mem.get_pos())

    # default to cap blocks in front of entity
    vec = agent.coordinate_transforms.look_vec(speaker_mem.yaw, speaker_mem.pitch)
    return cap * np.array(vec) + np.array(pos)
