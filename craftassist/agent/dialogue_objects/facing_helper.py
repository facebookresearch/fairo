"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from base_agent.base_util import ErrorWithResponse
from base_agent.dialogue_objects import interpret_relative_direction
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
            if d["relative_yaw"].get("angle"):
                return {"relative_yaw": int(d["relative_yaw"]["angle"])}
            elif d["relative_yaw"].get("yaw_span"):
                span = d["relative_yaw"].get("yaw_span")
                left = "left" in span or "leave" in span  # lemmatizer :)
                degrees = number_from_span(span) or 90
                if degrees > 0 and left:
                    return {"relative_yaw": -degrees}
                else:
                    return {"relative_yaw": degrees}
            else:
                pass
        elif d.get("relative_pitch"):
            if d["relative_pitch"].get("angle"):
                # TODO in the task make this relative!
                return {"relative_pitch": int(d["relative_pitch"]["angle"])}
            elif d["relative_pitch"].get("pitch_span"):
                span = d["relative_pitch"].get("pitch_span")
                down = "down" in span
                degrees = number_from_span(span) or 90
                if degrees > 0 and down:
                    return {"relative_pitch": -degrees}
                else:
                    return {"relative_pitch": degrees}
            else:
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
