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


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# FIXME temporary torch this, should not be here after june 2021
# will not do anything if a selector is already in place
# also doesn't convert locations properly, as LinearExtent is not nesting
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def convert_location_to_selector(lf):
    if type(lf) is not dict:
        return
    for k, v in lf.items():
        if k == "filters":
            if not v.get("selector"):
                if v.get("location", {}).get("reference_object"):
                    r = v["location"]["reference_object"]
                    selector_d = ref_obj_lf_to_selector({"reference_object": r})
                    del v["location"]
                    v["selector"] = selector_d
        else:
            convert_location_to_selector(v)


def maybe_listify_tags(t):
    return [t] if type(t) is str else t


def strip_prefix(s, pre):
    if s.startswith(pre):
        return s[len(pre) :]
    return s


# FIXME... be more careful with OR and NOT
def backoff_where(where_clause, triples_to_tags=True, canonicalize=True):
    """
    canonicalizes and flattens all triples in the "where_clause" subdict of filters_d,
    if triples_to_tags:  pulling just the obj_text if exists and if the pred_text is a "has_"
    if canonicalize: removing the word "the", setting to lowercase

    returns the list of obj_texts and modified dict
    """
    new_where_clause = deepcopy(where_clause)
    tags = []
    # doesn't check if where_clause is well formed
    conj = list(set(["AND", "OR", "NOT"]).intersection(set(where_clause.keys())))
    if conj:  # recurse
        conj = conj[0]
        for i in range(len(where_clause[conj])):
            clause_tags, new_subclause = backoff_where(
                where_clause[conj][i], triples_to_tags=triples_to_tags, canonicalize=canonicalize
            )
            tags.extend(clause_tags)
            new_where_clause[conj][i] = new_subclause
    else:  # a leaf
        if "input_left" in where_clause:
            # a triple expressed as a comparator
            if type(where_clause["input_left"]) is str and where_clause["input_left"].startswith(
                "has_"
            ):
                new_o = where_clause["input_right"]
                if type(new_o) is str:
                    if canonicalize:
                        new_o = strip_prefix(new_o, "the ")
                    tags.append(new_o)
                    if triples_to_tags:
                        new_where_clause["input_left"] = "has_tag"
                    new_where_clause["input_right"] = new_o
        else:  # triple...
            p = where_clause.get("pred_text")
            if p and type(p) is str and p.startswith("has_"):
                new_o = where_clause.get("obj_text")
                if new_o and type(new_o) is str:  # don't try to fix subqueries
                    if canonicalize:
                        new_o = strip_prefix(new_o, "the ")
                    tags.append(new_o)
                    if triples_to_tags:
                        new_where_clause["pred_text"] = "has_tag"
                    new_where_clause["obj_text"] = new_o
    return list(set(tags)), new_where_clause


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
