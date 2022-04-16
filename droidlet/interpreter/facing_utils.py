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


def interpret_relative_yaw(d):
    """
    converts the lf format into {"relative_yaw": <degrees>},
    where degrees is the amount the agent should turn from its current position.
    a left turn, or counterclockwise viewed from above has a negative degree value
    a right turn, or clockwise viewed from above has a positive degree value

    TODO specify this with a const to be imported
    """
    left = "left" in d["relative_yaw"] or "leave" in d["relative_yaw"]  # lemmatizer :)
    if left or "right" in d["relative_yaw"]:
        # don't allow negative values when a direction is specified, FIXME?
        degrees = abs(number_from_span(d["relative_yaw"])) or 90
        if left:
            return {"relative_yaw": degrees}
        else:
            return {"relative_yaw": -degrees}
    else:
        try:
            deg = int(d["relative_yaw"])
            return {"relative_yaw": deg}
        except:
            pass
