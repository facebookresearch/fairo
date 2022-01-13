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


# TODO: do this more carefully so that the mems are associated to
# their exact place in the logical form
def update_attended_and_link_lf(interpreter, mems):
    """
    for each mem in mems (a list of MemoryNodes), updates its attended time to now, 
    and links it to the interpreters logical form (if it has one).
    """
    interpreter.memory.update_recent_entities(mems)
    lf_memid = getattr(interpreter, "logical_form_memid", None)
    # a dummy interpreter may have no logical form memid associated to it...
    if lf_memid:
        for m in mems:
            interpreter.memory.add_triple(
                subj=m.memid, pred_text="attended_while_interpreting", obj=lf_memid
            )


# FIXME... be more careful with OR and NOT
def backoff_where(where_clause, triples_to_tags=True, canonicalize=True, lower=False):
    """
    util to canonicalize where clauses.
    if triples_to_tags:  flattens triples, pulling just the obj_text if exists and if the pred_text is a "has_"
    if canonicalize: removing the word "the"
    if lower: lowercases subj_text, obj_text, pred_text, value_left, and value_right

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
                where_clause[conj][i],
                triples_to_tags=triples_to_tags,
                canonicalize=canonicalize,
                lower=lower,
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
            if lower:
                if type(new_where_clause["input_right"]) is str:
                    new_where_clause["input_right"] = new_where_clause["input_right"].lower()
                if type(new_where_clause["input_left"]) is str:
                    new_where_clause["input_left"] = new_where_clause["input_left"].lower()
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
            if lower:
                for k in ["subj_text", "pred_text", "obj_text"]:
                    if new_where_clause.get(k) is str:
                        new_where_clause[k] = new_where_clause[k].lower()

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
