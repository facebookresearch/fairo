"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.dialog.dialogue_objects import interpret_relative_direction
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


class FacingInterpreter:
    def __call__(self, interpreter, speaker, d):
        # get these from memory, not player struct!!!!! FIXME!!!
        current_pitch = interpreter.agent.get_player().look.pitch
        current_yaw = interpreter.agent.get_player().look.yaw
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
            # TODO in the task use turn angle
            if "left" in d["relative_yaw"] or "right" in d["relative_yaw"]:
                left = "left" in span or "leave" in span  # lemmatizer :)
                degrees = number_from_span(span) or 90
                if degrees > 0 and left:
                    return {"relative_yaw": -degrees}
                else:
                    return {"relative_yaw": degrees}
            else:
                try:
                    deg = int(d["relative_yaw"])
                    return {"relative_yaw": deg}
                except:
                    pass
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
                    deg = int(d["relative_pitch"]["angle"])
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
