"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from base_agent.base_util import ErrorWithResponse
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


# TODO harmonize with MC... don't need this duplicated
class FacingInterpreter:
    def __call__(self, interpreter, speaker, d, head_or_body="head"):
        current_pitch = interpreter.agent.pitch
        if head_or_body == "head":
            current_yaw = interpreter.agent.pan
        else:
            current_yaw = interpreter.agent.base_yaw

        if d.get("yaw_pitch"):
            # make everything relative:
            span = d["yaw_pitch"]
            # for now assumed in (yaw, pitch) or yaw, pitch or yaw pitch formats
            yp = span.replace("(", "").replace(")", "").split()
            # negated in look_at in locobot_mover
            rel_yaw = current_yaw - float(yp[0])
            rel_pitch = current_pitch - float(yp[1])
            return {"yaw": rel_yaw, "pitch": rel_pitch}
        elif d.get("yaw"):
            # make everything relative:
            # for now assumed span is yaw as word or number
            w = float(word_to_num(d["yaw"].strip(" degrees").strip(" degree")))
            return {"yaw": current_yaw - w}
        elif d.get("pitch"):
            # make everything relative:
            # for now assumed span is pitch as word or number
            w = float(word_to_num(d["pitch"].strip(" degrees").strip(" degree")))
            return {"yaw": current_pitch - w}
        elif d.get("relative_yaw"):
            if d["relative_yaw"].get("angle"):
                return {"yaw": float(d["relative_yaw"]["angle"])}
            elif d["relative_yaw"].get("yaw_span"):
                span = d["relative_yaw"].get("yaw_span")
                left = "left" in span or "leave" in span  # lemmatizer :)
                degrees = number_from_span(span) or 90
                if degrees > 0 and left:
                    print(degrees)
                    return {"yaw": degrees}
                else:
                    print(-degrees)
                    return {"yaw": -degrees}
            else:
                pass
        elif d.get("relative_pitch"):
            if d["relative_pitch"].get("angle"):
                # TODO in the task make this relative!
                return {"pitch": int(d["relative_pitch"]["angle"])}
            elif d["relative_pitch"].get("pitch_span"):
                span = d["relative_pitch"].get("pitch_span")
                down = "down" in span
                degrees = number_from_span(span) or 90
                if degrees > 0 and down:
                    return {"pitch": -degrees}
                else:
                    return {"pitch": degrees}
            else:
                pass
        elif d.get("location"):
            loc_mems = interpreter.subinterpret["reference_locations"](
                interpreter, speaker, d["location"]
            )
            loc = loc_mems[0].get_pos()
            return {"target": loc}
        else:
            raise ErrorWithResponse("I am not sure where you want me to turn")
