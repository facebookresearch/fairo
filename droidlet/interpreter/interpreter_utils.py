"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from copy import deepcopy

SPEAKERLOOK = {"reference_object": {"special_reference": "SPEAKER_LOOK"}}
SPEAKERPOS = {"reference_object": {"special_reference": "SPEAKER"}}
AGENTPOS = {"reference_object": {"special_reference": "AGENT"}}


def ref_obj_lf_to_selector(ref_obj_dict):
    """ref_obj_dict should be {"reference_object ": ...}"""
    s = {
        "return_quantity": {
            "argval": {
                "ordinal": "FIRST",
                "polarity": "MIN",
                "quantity": {"attribute": {"linear_extent": {"source": ref_obj_dict}}},
            }
        }
    }
    return s


def maybe_listify_tags(t):
    return [t] if type(t) is str else t


def strip_prefix(s, pre):
    if s.startswith(pre):
        return s[len(pre) :]
    return s


# FIXME!? maybe start using triples appropriately now?
def tags_from_dict(filters_d):
    """
    flattens all triples in the "triples" subdict of filters_d,
    pulling just the obj_text if exists and if the pred_text is a "has_"
    """
    triples = filters_d.get("triples", [])
    tag_list = [
        strip_prefix(t.get("obj_text"), "the ")
        for t in triples
        if t.get("pred_text", "").startswith("has_")
    ]
    return list(set(tag_list))


def is_loc_speakerlook(d):
    # checks a location dict to see if it is SPEAKER_LOOK
    r = d.get("reference_object")
    if r and r.get("special_reference"):
        if r["special_reference"] == "SPEAKER_LOOK":
            return True
    return False


def process_spans_and_remove_fixed_value(d, original_words, lemmatized_words):
    """This function fetches the words corresponding to indices in the logical form
    and fetches the values of "fixed_value" key in the dictionary in place.
    """
    if type(d) is not dict:
        return
    for k, v in d.items():
        if type(v) == dict:
            # substitute the value of "fixed_value" in place in the dictionary
            if "fixed_value" in v.keys():
                d[k] = v["fixed_value"]
            else:
                process_spans_and_remove_fixed_value(v, original_words, lemmatized_words)
        elif type(v) == list and type(v[0]) == dict:
            # triples
            for a in v:
                process_spans_and_remove_fixed_value(a, original_words, lemmatized_words)
        else:
            try:
                sentence, (L, R) = v
                if sentence != 0:
                    raise NotImplementedError("Must update process_spans for multi-string inputs")
                if L > R:
                    L = R
                if L < 0:
                    L = 0
                if R > (len(lemmatized_words) - 1):
                    R = len(lemmatized_words) - 1
            except ValueError:
                continue
            except TypeError:
                continue
            original_w = " ".join(original_words[L : (R + 1)])
            # The lemmatizer converts 'it' to -PRON-
            if original_w == "it":
                d[k] = original_w
            else:
                d[k] = " ".join(lemmatized_words[L : (R + 1)])


#####FIXME!!!
# this is bad
# and
# in addition to being bad, abstraction is leaking
def coref_resolve(memory, d, chat):
    """Walk logical form "d" and replace coref_resolve values

    Possible substitutions:
    - a subdict lik SPEAKERPOS
    - a MemoryNode object
    - "NULL"

    Assumes spans have been substituted.
    """

    c = chat.split()
    if not type(d) is dict:
        return
    for k, v in d.items():
        if type(v) == dict:
            coref_resolve(memory, v, chat)
        if type(v) == list:
            for a in v:
                coref_resolve(memory, a, chat)
        v_copy = v
        if k == "location":
            # v is a location dict
            for k_ in v:
                if k_ == "contains_coreference":
                    v_copy = deepcopy(v)
                    val = SPEAKERPOS if "here" in c else SPEAKERLOOK
                    v_copy["reference_object"] = val["reference_object"]
                    v_copy["contains_coreference"] = "resolved"
            d[k] = v_copy
        elif k == "filters":
            # v is a reference object dict
            for k_ in v:
                if k_ == "contains_coreference":
                    v_copy = deepcopy(v)
                    if "this" in c or "that" in c:
                        v_copy["location"] = SPEAKERLOOK
                        v_copy["contains_coreference"] = "resolved"
                    else:
                        mems = memory.get_recent_entities("BlockObject")
                        if len(mems) == 0:
                            mems = memory.get_recent_entities(
                                "Mob"
                            )  # if its a follow, this should be first, FIXME
                            if len(mems) == 0:
                                v_copy[k_] = "NULL"
                            else:
                                v_copy[k_] = mems[0]
                        else:
                            v_copy[k_] = mems[0]
            d[k] = v_copy
        # fix/delete this branch! its for old broken spec
        else:
            for k_ in v:
                if k_ == "contains_coreference":
                    v_copy = deepcopy(v)
                    if "this" in c or "that" in c:
                        v_copy["location"] = SPEAKERLOOK
                        v_copy["contains_coreference"] = "resolved"
            d[k] = v_copy


if __name__ == "__main__":
    x = eval(
        "{'dialogue_type': 'PUT_MEMORY', 'filters': {'reference_object':{'contains_coreference': 'yes'}}, 'upsert': {'memory_data': {'memory_type': 'TRIPLE', 'has_tag': 'j'}}}"
    )
    y = coref_resolve(None, x, "that is a j")
