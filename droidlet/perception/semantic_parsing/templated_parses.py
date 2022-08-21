from word2number import w2n

color_words = [
    "blue",
    "red",
    "purple",
    "black",
    "green",
    "orange",
    "white",
    "gray",
    "yellow",
]

direction_words = {
    "top": "UP",
    "bottom": "DOWN",
    "above": "UP",
    "below": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "front": "FRONT",
    "back": "BACK",
    "behind": "BACK",
    "under": "DOWN",
}


def command(d):
    if type(d) is list:
        return {"dialogue_type": "HUMAN_GIVE_COMMAND", "event_sequence": d}
    else:
        return {"dialogue_type": "HUMAN_GIVE_COMMAND", "event_sequence": [d]}


def extract_where(sw, remove_s=False):
    where_clause = {}
    if len(sw) == 2:
        if len(sw[1]) > 0:
            if remove_s and sw[1][-1] == "s":
                x = sw[1][:-1]
            else:
                x = sw[1]
        if len(x) == 0:
            return {}
        if sw[0] in color_words:
            where_clause = {
                "where_clause": {
                    "AND": [
                        {"pred_text": "has_name", "obj_text": x},
                        {"pred_text": "has_colour", "obj_text": sw[0]},
                    ]
                }
            }
    if len(sw) == 1:
        if len(sw[0]) > 0:
            if remove_s and sw[0][-1] == "s":
                x = sw[0][:-1]
            else:
                x = sw[0]
        if len(x) == 0:
            return {}
        where_clause = {
            "where_clause": {
                "AND": [
                    {"pred_text": "has_name", "obj_text": x},
                ]
            }
        }
    return where_clause


def handle_move(w):
    """
    assumed w[0] == "move" or "go"
    """
    out = None
    reldir = None
    if len(w) < 4:
        return out
    mv_base = {"action_type": "MOVE", "location": {"reference_object": {}}}
    mv_base_between = {
        "action_type": "MOVE",
        "location": {"reference_object_1": {}, "reference_object_2": {}},
    }
    where_clause = {}
    wc1 = {}
    wc2 = {}
    if w[1] == "to" and (w[2] == "the" or w[2] == "a" or w[2] == "an"):
        if len(w) == 4:
            where_clause = extract_where(w[3:])  # go to the cow
        if len(w) == 5 and w[3] in color_words:
            where_clause = extract_where(w[3:])  # go to the red cow
        if len(w) == 7 and w[3] in direction_words.keys():
            where_clause = extract_where(w[6:])  # go to the left of the cow
            reldir = direction_words[w[3]]
        if (
            len(w) == 8 and w[3] in direction_words.keys() and w[6] in color_words
        ):  # go to the left of the red cow
            where_clause = extract_where(w[6:])
            reldir = direction_words[w[3]]
    if len(w) == 4 and w[1] in direction_words.keys():  # go above the cow
        where_clause = extract_where(w[3:])
        reldir = direction_words[w[1]]
    if (
        len(w) == 5 and w[1] in direction_words.keys() and w[3] in color_words
    ):  # go above the red cow
        where_clause = extract_where(w[3:])
        reldir = direction_words[w[1]]
    if len(w) == 7 and w[1] == "between":  # go between the x and the y
        wc1 = extract_where(w[3:4])
        wc2 = extract_where(w[6:])
        reldir = "between"
    if len(w) == 4 and w[1] == "between":  # go between the x
        where_clause = extract_where(w[3:], remove_s=True)
        reldir = "between"
    if len(w) == 4 and (w[1] == "inside" or w[1] == "in"):  # go in the x
        where_clause = extract_where(w[3:])
        reldir = "inside"
    if reldir and reldir == "between" and wc1 and wc2:
        mv_base_between["location"]["reference_object_1"] = {"filters": wc1}
        mv_base_between["location"]["reference_object_2"] = {"filters": wc2}
        return command(mv_base_between)
    if where_clause:
        mv_base["location"]["reference_object"] = {"filters": where_clause}
        if reldir:
            mv_base["location"]["relative_direction"] = reldir
        return command(mv_base)
    return None


# def handle_get(w):
#    """
#    assumed w[0] == "get" or "give" or "bring"
#    """
#    out = None
#    if len(w) < 4:
#        return out
#    mv_base = {"action_type": "MOVE",
#               "location": {"reference_object":{}}}
#    mv_base_between = {"action_type": "MOVE",
#                       "location": {"reference_object_1":{}, "reference_object_2":{}}}
#    where_clause = {}


def templated_match(chat):
    w = chat.split()
    if len(w) == 0:
        return None
    out = None
    if w[0] == "move" or w[0] == "go":
        return handle_move(w)
    if "that is a" in chat or "that is an" in chat or "this is a" in chat or "this is an" in chat:
        if len(w) == 4:
            out = {
                "dialogue_type": "put_memory",
                "filters": {"contains_coreference": "yes"},
                "upsert": {
                    "memory_data": {
                        "memory_type": "triple",
                        "triples": [{"pred_text": "has_tag", "obj_text": w[3]}],
                    }
                },
            }
        return out
    if w[0] == "destroy":
        if len(w) < 3:
            return None
        try:
            num = w2n.word_to_num(w[1])
        except:
            num = None
        if w[1] == "the":
            if len(w) == 3:
                where_clause = {
                    "where_clause": {
                        "AND": [
                            {"pred_text": "has_name", "obj_text": w[2]},
                        ]
                    }
                }
            if len(w) == 4 and w[2] in color_words:
                where_clause = {
                    "where_clause": {
                        "AND": [
                            {"pred_text": "has_name", "obj_text": w[3]},
                            {"pred_text": "has_colour", "obj_text": w[2]},
                        ]
                    }
                }
            out = command(
                {"action_type": "DESTROY", "reference_object": {"filters": where_clause}},
            )
            return out
        if num is not None:
            f = {
                "selector": {
                    "same": "disallowed",
                    "return_quantity": "random",
                    "ordinal": str(num),
                },
            }
            if len(w) == 3:
                f["where_clause"] = {
                    "AND": [
                        {"pred_text": "has_name", "obj_text": w[2]},
                    ]
                }
            out = command(
                {"action_type": "DESTROY", "reference_object": {"filters": f}},
            )
            return out

    if w[0] == "build":
        if len(w) > 2:
            try:
                num = w2n.word_to_num(w[1])
            except:
                num = None
        if num is not None:
            f = {
                "selector": {"same": "allowed", "return_quantity": "random", "ordinal": str(num)},
            }
            if len(w) == 3:
                f["where_clause"] = {
                    "AND": [
                        {"pred_text": "has_name", "obj_text": w[2]},
                    ]
                }
            out = command(
                {"action_type": "BUILD", "schematic": {"filters": f}},
            )
            return out


if __name__ == "__main__":
    p = templated_match("destroy two houses")
    q = templated_match("go to the left of the house")
    k = templated_match("go between the j")
