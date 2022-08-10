"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import math
from droidlet.shared_data_structs import ErrorWithResponse
from .interpreter_utils import SPEAKERLOOK, backoff_where, update_attended_and_link_lf


def interpret_relative_direction(interpreter, location_d):
    steps = location_d.get("steps", None)
    if steps is not None:
        try:
            steps = math.ceil(float(steps))
        except:
            steps = None
    reldir = location_d.get("relative_direction")
    return steps, reldir


class ReferenceLocationInterpreter:
    def __call__(self, interpreter, speaker, d):
        """
        Location dict -> ref obj memories.
        Side effect: adds mems to agent_memory.recent_entities

        args:
        interpreter:  root interpreter.
        speaker (str): The name of the player/human/agent who uttered
            the chat resulting in this interpreter
        d: logical form from semantic parser
        """
        loose_speakerlook = False
        expected_num = 1
        # FIXME! merge/generalize this, code in spatial_reasoning, and filter_by_sublocation
        if d.get("relative_direction") == "BETWEEN":
            loose_speakerlook = True
            expected_num = 2
            ref_obj_1 = d.get("reference_object_1")
            ref_obj_2 = d.get("reference_object_2")
            if ref_obj_1 and ref_obj_2:
                interpreter.allow_clarificaiton = False
                mem1 = interpreter.subinterpret["reference_objects"](
                    interpreter,
                    speaker,
                    ref_obj_1,
                    loose_speakerlook=loose_speakerlook,
                )[0]
                mem2 = interpreter.subinterpret["reference_objects"](
                    interpreter,
                    speaker,
                    ref_obj_2,
                    loose_speakerlook=loose_speakerlook,
                )[0]
                if mem1 is None or mem2 is None:
                    raise ErrorWithResponse("I don't know what you're referring to")
                mems = [mem1, mem2]
                update_attended_and_link_lf(interpreter, mems)
                return mems

            else:
                interpreter.allow_clarification = False

        default_loc = getattr(interpreter, "default_loc", SPEAKERLOOK)
        ref_obj = d.get("reference_object", default_loc["reference_object"])
        mems = interpreter.subinterpret["reference_objects"](
            interpreter,
            speaker,
            ref_obj,
            loose_speakerlook=loose_speakerlook,
        )

        # FIXME use FILTERS here!!
        if len(mems) < expected_num:
            w = refobj.get("filters", {}).get("where_clause")
            if w:
                tags, _ = backoff_where(w)
                tags = set(tags)
                for memtype in interpreter.workspace_memory_prio:
                    cands = interpreter.memory.get_recent_entities(memtype)
                    mems = [c for c in cands if any(set.intersection(set(c.get_tags()), tags))]
                    if len(mems) >= expected_num:
                        break

        if len(mems) < expected_num:
            raise ErrorWithResponse("I don't know what you're referring to")

        # FIXME:
        mems = mems[:expected_num]
        update_attended_and_link_lf(interpreter, mems)

        return mems
