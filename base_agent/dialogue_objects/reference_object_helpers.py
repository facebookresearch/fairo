"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import re
import numpy as np
from typing import cast, List, Tuple, Dict


from dialogue_object_utils import SPEAKERLOOK
from dialogue_object import ConfirmReferenceObject
from filter_helper import FilterInterpreter
from location_helpers import interpret_relative_direction
from base_agent.base_util import euclid_dist
from base_agent.memory_attributes import LookRayDistance, LinearExtentAttribute
from base_agent.memory_nodes import ReferenceObjectNode
from base_agent.base_util import T, XYZ, ErrorWithResponse, NextDialogueStep


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
    """ make a search dictionary for a BasicMemorySearcher to return the special ReferenceObject"""
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
    """ subinterpret a special reference object.  
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


# TODO rewrite functions in intepreter and helpers as classes
# finer granularity of (code) objects
# interpreter is an input to interpret ref object, maybe clean that up?
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
    limit=1,
    loose_speakerlook=False,
    allow_clarification=True,
) -> List[ReferenceObjectNode]:
    """this tries to find a ref obj memory matching the criteria from the
    ref_obj_dict

    args:
    interpreter:  root interpreter.
    speaker (str): The name of the player/human/agent who uttered 
        the chat resulting in this interpreter
    d: logical form from semantic parser

    extra_tags (list of strings): tags added by parent to narrow the search
    limit (natural number): maximum number of reference objects to return
    allow_clarification (bool): should a Clarification object be put on the DialogueStack    
    """
    F = d.get("filters")
    special = d.get("special_reference")
    # F can be empty...
    assert (F is not None) or special, "no filters or special_reference sub-dicts {}".format(d)
    if special:
        mem = get_special_reference_object(interpreter, speaker, special)
        return [mem]

    if F.get("contains_coreference", "NULL") != "NULL":
        mem = F["contains_coreference"]
        if isinstance(mem, ReferenceObjectNode):
            return [mem]
        elif mem == "resolved":
            pass
        else:
            logging.error("bad coref_resolve -> {}".format(mem))

    if len(interpreter.progeny_data) == 0:
        # FIXME when tags from dict is better or removed entirely...
        F["has_extra_tags"] = F.get("has_extra_tags", []) + extra_tags
        # TODO Add ignore_player maybe?
        candidate_mems = apply_memory_filters(interpreter, speaker, F)
        if len(candidate_mems) > 0:
            # FIXME?
            candidates = [(c.get_pos(), c) for c in candidate_mems]
            r = filter_by_sublocation(
                interpreter, speaker, candidates, d, limit=limit, loose=loose_speakerlook
            )
            return [mem for _, mem in r]
        elif allow_clarification:
            # no candidates found; ask Clarification
            # TODO: move ttad call to dialogue manager and remove this logic
            interpreter.action_dict_frozen = True
            confirm_candidates = apply_memory_filters(interpreter, speaker, F)
            objects = object_looked_at(interpreter.agent, confirm_candidates, speaker=speaker)
            if len(objects) == 0:
                raise ErrorWithResponse("I don't know what you're referring to")
            _, mem = objects[0]
            interpreter.provisional["object_mem"] = mem
            interpreter.provisional["F"] = F
            interpreter.dialogue_stack.append_new(ConfirmReferenceObject, mem)
            raise NextDialogueStep()
        else:
            raise ErrorWithResponse("I don't know what you're referring to")

    else:
        # clarification answered
        r = interpreter.progeny_data[-1].get("response")
        if r == "yes":
            # TODO: learn from the tag!  put it in memory!
            return [interpreter.provisional.get("object_mem")] * limit
        else:
            raise ErrorWithResponse("I don't know what you're referring to")


def apply_memory_filters(interpreter, speaker, filters_d) -> List[ReferenceObjectNode]:
    """Return a list of (xyz, memory) tuples encompassing all possible reference objects"""
    FI = FilterInterpreter()
    F = FI(interpreter, speaker, filters_d)
    memids, _ = F()
    mems = [interpreter.agent.memory.get_mem_by_id(i) for i in memids]
    #    f = {"triples": [{"pred_text": "has_tag", "obj_text": tag} for tag in tags]}
    #    mems = interpreter.memory.basic_search(f)
    return mems


# FIXME make me a proper filters object
# TODO filter by INSIDE/AWAY/NEAR
def filter_by_sublocation(
    interpreter,
    speaker,
    candidates: List[Tuple[XYZ, T]],
    d: Dict,
    limit=1,
    all_proximity=10,
    loose=False,
) -> List[Tuple[XYZ, T]]:
    """Select from a list of candidate (xyz, object) tuples given a sublocation

    If limit == 'ALL', return all matching candidates

    Returns a list of (xyz, mem) tuples
    """
    F = d.get("filters")
    assert F is not None, "no filters: {}".format(d)
    default_loc = getattr(interpreter, "default_loc", SPEAKERLOOK)
    location = F.get("location", default_loc)
    #    if limit == 1:
    #        limit = get_repeat_num(d)

    reldir = location.get("relative_direction")
    if reldir:
        if reldir == "INSIDE":
            # FIXME formalize this better, make extensible
            if location.get("reference_object"):
                # should probably return from interpret_reference_location...
                ref_mems = interpret_reference_object(
                    interpreter, speaker, location["reference_object"]
                )
                for l, candidate_mem in candidates:
                    I = interpreter.agent.on_demand_perception.get["check_inside"]
                    if I:
                        if I([candidate_mem, ref_mems[0]]):
                            return [(l, candidate_mem)]
                    else:
                        raise ErrorWithResponse("I don't know how to check inside")
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
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
        else:
            # FIXME need some tests here
            # reference object location, i.e. the "X" in "left of X"
            mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location)

            # FIXME!!! handle frame better, might want agent's frame instead
            eid = interpreter.agent.memory.get_player_by_name(speaker).eid
            self_mem = interpreter.agent.memory.get_mem_by_id(interpreter.agent.memory.self_memid)
            L = LinearExtentAttribute(
                interpreter.agent, {"frame": eid, "relative_direction": reldir}, mem=self_mem
            )
            proj = L([c[1] for c in candidates])

            # filter by relative dir, e.g. "left of Y"
            proj_cands = [(p, c) for (p, c) in zip(proj, candidates) if p > 0]

            # "the X left of Y" = the right-most X that is left of Y
            if limit == "ALL":
                limit = len(proj_cands)
            return [c for (_, c) in sorted(proj_cands, key=lambda p: p[0])][:limit]
    else:
        # no reference direction: choose the closest
        mems = interpreter.subinterpret["reference_locations"](interpreter, speaker, location)
        steps, reldir = interpret_relative_direction(interpreter, d)
        ref_loc, _ = interpreter.subinterpret["specify_locations"](
            interpreter, speaker, mems, steps, reldir
        )
        if limit == "ALL":
            return list(filter(lambda c: euclid_dist(c[0], ref_loc) <= all_proximity, candidates))
        else:
            candidates.sort(key=lambda c: euclid_dist(c[0], ref_loc))
            return candidates[:limit]
    return []  # this fixes flake but seems awful?


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
