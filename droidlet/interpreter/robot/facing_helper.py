"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from droidlet.shared_data_structs import ErrorWithResponse
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
        self_mem = interpreter.memory.get_mem_by_id(interpreter.memory.self_memid)
        current_yaw, current_pitch = self_mem.get_yaw_pitch()

        # FIXME!!!! make a proper PoseNode
        # if head_or_body == "head":
        #    current_yaw = interpreter.agent.pan
        # else:
        #    current_yaw = interpreter.agent.base_yaw

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
            if any(x in d["relative_yaw"] for x in ["left", "leave", "right"]):
                left = "left" in d["relative_yaw"] or "leave" in d["relative_yaw"]  # lemmatizer :)
                degrees = number_from_span(d["relative_yaw"]) or 90
                # these are different than mc for no reason...? mc uses relative_yaw, these use yaw
                if degrees > 0 and left:
                    return {"yaw": -degrees}
                else:
                    return {"yaw": degrees}
            else:
                try:
                    deg = int(word_to_num(d["relative_yaw"].strip(" degree")))
                    return {"yaw": -deg if deg > 0 else deg} 
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
                    deg = int(number_from_span(d["relative_pitch"]))
                    return {"relative_pitch": deg}
                except:
                    pass
        elif d.get("location"):
            loc_mems = interpreter.subinterpret["reference_locations"](
                interpreter, speaker, d["location"]
            )
            loc = loc_mems[0].get_pos()
            return {"target": loc}
        else:
            raise ErrorWithResponse("I am not sure where you want me to turn")
