"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import numpy as np
from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.interpreter import interpret_relative_direction
from droidlet.interpreter.facing_utils import number_from_span, interpret_relative_yaw
from word2number.w2n import word_to_num


def number_from_span(span):
    # this will fail in many cases....
    words = span.split()
    degrees = None
    for w in words:
        try:
            degrees = int(w)
        except:
            pass
    if not degrees:
        try:
            degrees = word_to_num(span)
        except:
            pass
    return degrees


# WARNING: everything here is in degrees.
# TODO: change fixed values in DSL to be radians
class FacingInterpreter:
    def __call__(self, interpreter, speaker, d):
        self_mem = interpreter.memory.get_mem_by_id(interpreter.memory.self_memid)
        current_yaw, current_pitch = self_mem.get_yaw_pitch()
        # WARNING:
        current_yaw = np.rad2deg(current_yaw)
        current_pitch = np.rad2deg(current_pitch)
        if d.get("yaw_pitch"):
            span = d["yaw_pitch"]
            # for now assumed in (yaw, pitch) or yaw, pitch or yaw pitch formats
            yp = span.replace("(", "").replace(")", "").split()
            return {"head_yaw_pitch": (int(yp[0]), int(yp[1]))}
        elif d.get("yaw"):
            # for now assumed span is yaw as word or number
            w = d["yaw"].strip(" degrees").strip(" degree")
            return {"head_yaw_pitch": (word_to_num(w), current_pitch)}
        elif d.get("pitch"):
            # for now assumed span is pitch as word or number
            w = d["pitch"].strip(" degrees").strip(" degree")
            return {"head_yaw_pitch": (current_yaw, word_to_num(w))}
        elif d.get("relative_yaw"):
            return interpret_relative_yaw(d)
        elif d.get("relative_pitch"):
            if "down" in d["relative_pitch"] or "up" in d["relative_pitch"]:
                down = "down" in d["relative_pitch"]
                degrees = number_from_span(d["relative_pitch"]) or 90
                if degrees > 0 and down:
                    return {"relative_pitch": -degrees}
                else:
                    return {"relative_pitch": degrees}
            else:
                # TODO in the task make this relative!
                try:
                    deg = int(number_from_span(d["relative_pitch"]))
                    return {"relative_pitch": deg}
                except:
                    pass
        elif d.get("location"):
            mems = interpreter.subinterpret["reference_locations"](
                interpreter, speaker, d["location"]
            )
            steps, reldir = interpret_relative_direction(interpreter, d["location"])
            loc, _ = interpreter.subinterpret["specify_locations"](
                interpreter, speaker, mems, steps, reldir
            )
            return {"head_xyz": loc}
        else:
            raise ErrorWithResponse("I am not sure where you want me to turn")
