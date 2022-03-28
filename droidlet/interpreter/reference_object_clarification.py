import logging
import ipdb
from droidlet.memory.memory_nodes import (
    LocationNode,
    PlayerNode,
    ReferenceObjectNode,
    SelfNode,
    TaskNode,
)
from droidlet.interpreter.interpret_reference_objects import filter_by_sublocation
from droidlet.shared_data_structs import ErrorWithResponse

DISALLOWED_REF_OBJS = (LocationNode, SelfNode, PlayerNode)

def clarify_reference_objects(interpreter, speaker, d, candidate_mems, num_refs):
    ipdb.set_trace(context=5)

    # Decide what clarification class we're in
    if num_refs < 1:
        logging.error("There appear to be no references in the command to clarify, debug")
        raise ErrorWithResponse("I don't know what you're referring to")
    if len(candidate_mems) == 0:
        clarification_class = "REF_NO_MATCH"
        dlf = cc1_to_dlf(interpreter, speaker, d, clarification_class)
    elif num_refs > len(candidate_mems):
        clarification_class = "REF_TOO_FEW"
    else:
        clarification_class = "REF_TOO_MANY"

def cc1_to_dlf(interpreter, speaker, d, clarification_class):
    # No reference objects found, expand search and ask for clarification
    clarification_query = "SELECT MEMORY FROM ReferenceObject WHERE x>-1000"
    _, clarification_ref_obj_mems = interpreter.memory.basic_search(clarification_query)
    objects = [x for x in clarification_ref_obj_mems if not isinstance(x, DISALLOWED_REF_OBJS)]
    mems = filter_by_sublocation(
        interpreter,
        speaker,
        objects,
        d,
    )
    if len(mems) == 0:
        raise ErrorWithResponse("I don't know what you're referring to")
    else:
        # Build the matchind dialog DLF
        action = retrieve_action_dict(interpreter, d)
        if not action:
            logging.error("Unable to retrieve a matching action dictionary from top level logical form.")
            raise ErrorWithResponse("I don't know what you're referring to")
        
        return build_dlf(clarification_class, mems, action)

def retrieve_action_dict(interpreter, d):
    action = False
    potential_actions = interpreter.logical_form["action_sequence"]
    for pa in potential_actions:
        if pa["reference_object"] == d:
            action = pa
    return action

def build_dlf(cc, candidates, action):
    dlf = {
        "dialogue_type": 'CLARIFICATION',
        "action": action,
        "class": {
            "error_type": cc,
            "candidates": candidates,
        }
    }
    return dlf