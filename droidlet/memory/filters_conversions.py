from copy import deepcopy
import json

FILTERS_KW = ["SELECT", "FROM", "WHERE", "ORDER BY", "LIMIT", "SAME", "CONTAINS_COREFERENCE"]
LIMITS = {"FIRST": "1", "SECOND": "2", "THIRD": "3"}

# name resolution for properties:  DOES NOT EXIST
# that is: "triples"/"properties"/"column names" are equivalent, and a
# single memory is allowed to have multiple values for a single property!!!
# this sounds weird when you think about many standard table columns, but
# for others its natural, and certainly for Triples its natural...

##################################################
# string utils:
##################################################


def match_symbol(text, pidx=0, s=("(", ")")):
    """
    given an opening and closing pair of symbols
    defaulting to "(" and ")",
    returns the index in the text str where (the start of)
    a closing instance of the pair matches the first opening
    instance of the pair at or after idx.
    if the pair is umnmatched, returns -1
    """
    assert s[0] != s[1]
    opened = False
    open_count = 0
    i = pidx
    L = len(text)
    while i < L:
        if text[i : min(i + len(s[0]), L)] == s[0]:
            open_count += 1
            opened = True
            i = i + len(s[0])
        elif text[i : min(i + len(s[1]), L)] == s[1] and opened:
            open_count -= 1
            if open_count == 0:
                return i
            i = i + len(s[1])
        else:
            i += 1

    return -1


def remove_enclosing_symbol(text, s=("(", ")")):
    """
    remove a matched pair of symbols enclosing a string
    i.e. "(some text)" --> "some text"
    """
    if text[0 : len(s[0])] == s[0]:
        c = match_symbol(text, s=s)
        if len(text) == c + len(s[1]):
            text = text[len(s[0]) : -len(s[1])]
    return text


def remove_nested_enclosing_symbol(text, s=("(", ")")):
    """
    clean nested matched pairs of symbols and spaces
    enclosing a string
    i.e. "(( some text) )" --> "some text"
    """
    text = text.strip()
    while remove_enclosing_symbol(text, s=s) != text:
        text = remove_enclosing_symbol(text)
        text = text.strip()
    return text


def find_keyword(S, start=0, keywords=FILTERS_KW):
    # find the first keyword present in the string
    for kw in keywords:
        kidx = S.find(kw, start)
        if kidx >= 0:
            break
    return kidx


def maybe_eval_literal(clause):
    try:
        output = json.loads(clause)
    except:
        output = clause
    if type(output) is tuple:
        output = output[0]
    return output


##################################################
# conversion from dict to str:
##################################################


def new_filters_to_sqly(d):
    """
    Takes a query in dictionary form with keys:
    "output": corresponding to the "SELECT" clause; with string values "COUNT" or "MEMORY"
        or attribute dict as possible values
    "memory_type": corresponding to "FROM"; should be a MemoryNode type
    "where_clause":  a tree of dicts where sentences (lists)
        of clauses are keyed by a conjunction.  leaves in the tree are
        comparators.  to represent a kb triple, use a comparator with
        input left being the pred_text, and input_right the obj memid or obj_text
        if obj memid, use "MEMID_EQUAL" as the equality type; otherwise use "EQUAL"
    "selector": corresponding to "ORDER BY", "LIMIT", "SAME"
    "contains_coreference": corresponding to "CONTAINS_COREFERENCE"

    returns a string of the following form:

    SELECT <attribute>;
    FROM mem_type(s);
    WHERE <sentence of clauses>;
    ORDER BY <attribute>;
    LIMIT <ordinal> DESC/ASC;
    SAME ALLOWED/DISALLOWED/REQUIRED;
    CONTAINS_COREFERENCE;

    FIXME!! TODO !! spec for subqueries in comparators and attributes, ...
    """
    S = "SELECT "
    o = d.get("output", "MEMORY")
    if o == "MEMORY" or o == "COUNT":
        S = S + o + " "
    else:
        if o.get("attribute") is None:
            raise Exception("malformed output dict {}".format(o))
        attrs = o["attribute"]
        if type(attrs) is list and len(attrs) > 1:
            attrs_str = " ( "
        else:
            attrs_str = " "
            attrs = [attrs]
        for i in range(len(attrs)):
            a = str(attrs[i])
            # FIXME do this more reliably
            if (
                a[0] == "{"
                or len(a.split()) > 1
                or any([k in a for k in ["SELECT", "WHERE", "ORDER BY"]])
            ):
                a = "(" + a + ")"
            attrs_str = attrs_str + a + " "
            if i < len(attrs):
                attrs_str = attrs_str + ", "
        if len(attrs) > 1:
            attrs_str = attrs_str + " ) "
        S = S + attrs_str
    if d.get("memory_type"):
        S = S + "FROM " + d["memory_type"] + "; "
    if d.get("where_clause"):
        S = S + "WHERE " + sqlyify_where_clause(d["where_clause"]) + "; "
    if d.get("selector"):
        return_q = d["selector"].get("return_quantity")
        limit = d["selector"].get("ordinal") or "1"
        if LIMITS.get(limit):
            limit = LIMITS[limit]
        if return_q:
            if return_q == "random":
                S = S + " ORDER BY RANDOM LIMIT " + limit + " "
            elif return_q.get("argval"):
                S = (
                    S
                    + " ORDER BY ("
                    + str(return_q["argval"]["quantity"]["attribute"])
                    + "); LIMIT "
                )
                S = S + limit + " "
                S = S + {"MAX": "DESC", "MIN": "ASC"}[return_q["argval"]["polarity"]]
        elif d["selector"].get("location"):
            S = S + "ORDER BY LOCATION (" + str(d["selector"]["location"]) + "); "
        if d["selector"].get("same"):
            S = S + "; SAME " + selector["same"]
    if d.get("contains_coreference"):
        S = S + "CONTAINS_COREFERENCE " + d["contains_coreference"] + ";"
    return S


def get_inequality_symbol(iq):
    if iq == "GREATER_THAN":
        return ">"
    elif iq == "LESS_THAN":
        return "<"
    elif iq == "GREATER_THAN_EQUAL":
        return ">="
    elif iq == "LESS_THAN_EQUAL":
        return "<="
    # FIXME deprecate this, just do NOT in an EQUAL
    elif iq == "NOT_EQUAL":
        return "!="
    elif iq == "EQUAL":
        return "="
    # FIXME deprecate this, better syntax for triples and memids
    # use . notation? $variable.memid?
    elif iq == "MEMID_EQUAL":
        return "=#="
    else:
        if type(iq) is dict:
            assert iq.get("close_tolerance") or iq.get("modulus")
            eq = "="
            if iq.get("modulus"):
                eq = "%_({})".format(iq["modulus"])
            if iq.get("close_tolerance"):
                return eq + "(+-{})".format(iq["close_tolerance"])
            else:
                return eq


def sqlyify_where_clause(c):
    """
    c should be a dict of of the (recursive) form {"AND"/"OR"/"NOT": [clause_0, ... , clause_m]},
    where each clause_i in the list either has the same form or is a comparator
    if "NOT", should be {"NOT": [clause]} (a single entry in the list)
    """
    # FIXME ANY/ALL
    for k, v in c.items():
        clause_texts = []
        assert len(v) > 0
        assert type(v) is list
        if k == "NOT":
            assert len(v) == 1
        for clause in v:
            if clause.get("input_left"):
                input_left = clause["input_left"]["value_extractor"]
                if input_left.get("attribute"):
                    input_left = str(input_left["attribute"])
                else:
                    input_left = str(input_left)
                input_right = str(clause["input_right"]["value_extractor"])
                inequality_symbol = get_inequality_symbol(clause["comparison_type"])
                s = input_left + " " + inequality_symbol + " " + input_right
                if clause.get("comparison_measure"):
                    s = s + " MEASURED_IN " + clause["comparison_measure"] + " "
            elif clause.get("pred_text"):
                if clause.get("subj"):
                    s = "<< #{}, {}, ? >>".format(clause["subj"], clause["pred_text"])
                elif clause.get("obj"):
                    s = "<< ?, {}, #{} >>".format(clause["pred_text"], clause["obj"])
                else:
                    s = "<< ?, {}, {} >>".format(clause["pred_text"], clause["obj_text"])
            else:
                s = sqlyify_where_clause(clause)
            clause_texts.append(s)
        if k == "NOT":
            assert len(clause_texts) == 1
            return "( NOT " + clause_texts[0] + " ) "
        else:
            return "(" + (" " + k + " ").join(clause_texts) + ")"


##################################################
# conversion from str to dict:
##################################################


def sqly_to_new_filters(S):
    """
    Basic form:

    SELECT <attribute>;
    FROM mem_type(s);
    WHERE <sentence of clauses>;
    ORDER BY <attribute>;
    LIMIT <ordinal> DESC/ASC;
    SAME ALLOWED/DISALLOWED/REQUIRED;
    CONTAINS_COREFERENCE ;

    for now it assumed that <attribute> is either a string, or if it is more complex
    than a single string, it is enclosed in ().
    the WHERE clause is assumed enclosed in parens.
    the <attribute> in the ORDER BY clause is again either string or enclosed in {}.
    """
    # TODO/FIXME:
    # subqueries for left or right values and in other places
    #     These can currently be done using FILTERS dicts
    # special notation/equality in comparators for obj or obj_text search etc...
    #     These can be done by hand using SELECT obj FROM triples WHERE ... etc.
    #     but this breaks the abstraction somewhat

    blocks = split_sqly(S)
    d = {}
    for b in blocks:
        idx = -1
        for kw in FILTERS_KW:
            idx = b.find(kw)
            if idx == 0:
                clause = b[len(kw) + 1 :]
                break
        assert idx > -1
        {
            "SELECT": convert_output_from_sqly,
            "FROM": convert_memtype_from_sqly,
            "WHERE": convert_where_from_sqly,
            "ORDER BY": convert_order_by_from_sqly,
            "LIMIT": convert_limit_from_sqly,
            "SAME": convert_same_from_sqly,
            "CONTAINS_COREFERENCE": convert_coref_from_sqly,
        }[kw](clause, d)

    return d


def find_next_block(S, keywords=FILTERS_KW):
    """
    tries to find the next block of statements.
    if an open paren or open brace occurs before the keyword,
    the keyword might be part of a subquery.
    """
    kidx = find_keyword(S, keywords=keywords)
    if kidx == 0:
        first_space = S.find(" ")
        if first_space > 0:
            kidx = find_keyword(S[first_space:], keywords=keywords)
            if kidx > -1:
                kidx = kidx + first_space
    pidx = S.find("(")
    if (kidx < pidx and kidx > 0) or pidx < 0:
        # no parens, current block ends at keyword
        return kidx
    else:
        # parens, there might be a child filter inside.
        # find the close parens and that is end of block
        pidx = match_symbol(S)
        if pidx > 0:
            return find_keyword(S, start=pidx, keywords=keywords)
        else:
            return -1


def split_sqly(S, keywords=FILTERS_KW):
    """
    splits a sqly statement into blocks
    the blocks either start with one of the strings in keywords
    or start after an outermost matching (closing) paren
    Only does one level of split...
    """
    # sanity check
    p = match_symbol(S)
    if p < 0 and "(" in S:
        raise Exception("query {} has unbalanced parens".format(S))
    s = S
    clauses = []
    idx = 0
    while True:
        idx = find_next_block(s, keywords=keywords)
        if idx > 0:
            clauses.append(s[:idx].strip().strip(";"))
            s = s[idx:]
        else:
            clauses.append(s.strip().strip(";"))
            break
    return clauses


def treeify_sqly_where(clause):
    """
    converts a where clause in sqly form to a nested dict:
    for example:
    (has_name = cow AND (has_colour = green OR has_colour = red))
    -->
    {'AND': ['has_name = cow', {'OR': ['has_colour = green', 'has_colour = red']}]}
    """
    clause = remove_nested_enclosing_symbol(clause)
    t = split_sqly(clause, keywords=["AND", "OR"])
    if len(t) == 0:
        raise Exception("empty clause")
    if len(t) == 1:
        # either a leaf clause or a "NOT"
        if t[0][:3] == "NOT":
            return {"NOT": treeify_sqly_where(t[0][4:])}
        else:
            return t[0]
    conj_list = {"AND": False, "OR": False}
    for i in range(len(t)):
        c = t[i]
        if c[:3] == "AND":
            conj_list["AND"] = True
            t[i] = c[3:].strip()
        elif c[:2] == "OR":
            conj_list["OR"] = True
            t[i] = c[2:].strip()
    if conj_list["OR"] and conj_list["AND"]:
        raise Exception("AND and OR at same level of clause {}".format(clause))
    if not (conj_list["OR"] or conj_list["AND"]):
        raise Exception("multiple blocks in clause but no conjunctions {}".format(clause))
    if conj_list["OR"]:
        return {"OR": [treeify_sqly_where(c) for c in t]}
    else:
        return {"AND": [treeify_sqly_where(c) for c in t]}


def convert_where_tree(where_tree):
    """
    converts a treeified where tree (output from treeify_sqly_where)
    into a new-style FILTERS where clause by recursively converting
    clauses into comparators
    """
    if type(where_tree) is str:
        where_tree = remove_nested_enclosing_symbol(where_tree)
        if where_tree[0] == "<":
            return triple_str_to_dict(where_tree)
        else:
            return where_leaf_to_comparator(where_tree)
    output = {}
    if where_tree.get("NOT") and type(where_tree["NOT"]) is str:
        output["NOT"] = [where_leaf_to_comparator(where_tree["NOT"])]
        return output
    for k, v in where_tree.items():
        if k in ["AND", "OR", "NOT"]:
            # a subtree
            assert type(v) is list
            output[k] = [convert_where_tree(t) for t in v]
    return output


def triple_str_to_dict(clause):
    """
    converts a triple (for a where_clause) in the form
    <<#subj, pred_text, #obj/obj_text>>
    to dictionary form. it assumed that one of the three entries is
    replaced by a "?"
    if the obj memid is fixed (as opposed to the obj_text),
    use a "#" in front of the memid.  subj_text is not a valid
    possibility for the first entry of the triple; still, if a query uses
    a fixed subj, it should be preceded with a "#".
    the order is assumed to be subj, pred, obj.
    examples:

    "find me a record whose name is bob":
    << ?, has_name, bob >> --> {"pred_text": "has_name", "obj_text": "bob"}

    "find me a record who is a friend of the entity with memid
    dd2ca5a4c5204fc09c71279f8956a2b1":
    << ?, friend_of, #dd2ca5a4c5204fc09c71279f8956a2b1 >>  -->
          {"pred_text": "friend_of", "obj": "dd2ca5a4c5204fc09c71279f8956a2b1"}

    "find me a record x for which the entity with memid
    dd2ca5a4c5204fc09c71279f8956a2b1" is a parent_of x:
    << #dd2ca5a4c5204fc09c71279f8956a2b1, parent_of, ? >>  -->
          {"pred_text": "parent_of", "subj": "dd2ca5a4c5204fc09c71279f8956a2b1"}

    TODO:
    This does not currently handle nested queries.
    This does not currently handle multiple "?"
    """
    terms = remove_enclosing_symbol(clause, ("<<", ">>")).split(",")
    terms = [t.strip() for t in terms]
    assert terms[1] and terms[1] != "?"
    out = {"pred_text": terms[1]}
    if terms[0] == "?":
        if terms[2] == "?":
            raise Exception(
                "queries with both subj and obj unfixed in a triple are not yet supported"
            )
        assert terms[2] != "?"
        if terms[2][0] == "#":
            out["obj"] = terms[2][1:]
        else:
            out["obj_text"] = terms[2]
    else:
        if terms[0][0] == "#":
            out["subj"] = terms[0][1:]
        else:
            raise Exception(
                'queries with a "subj_text" (as opposed to subj memid) in a triple are not supported'
            )
    return out


def where_leaf_to_comparator(clause):
    """
    converts a leaf in sqly clause into a FILTERs comparator
    for example
    'has_name = cow'
    -->
    {"input_left": {"value_extractor": "has_name"},
     "input_right": {"value_extractor": "cow"},
     "comparison_type": "EQUAL"}
    """
    # TODO having_measure
    eq_idx = clause.find("=")
    memideq_idx = clause.find(
        "=#="
    )  # special equality for memids instead of subject or object _text_
    lt_idx = clause.find("<")
    lte_idx = clause.find("<=")
    gt_idx = clause.find(">")
    gte_idx = clause.find(">=")
    mod_idx = clause.find("%")
    # everything will break if clause is complicated enough that it has internal comparators FIXME?
    # not obvious we should be converting those back forth though
    assert not (gt_idx > -1 and lt_idx > -1)
    assert not (eq_idx > -1 and mod_idx > -1)
    if lt_idx > -1:
        eq_idx = -1
        left_text = clause[:lt_idx]
        if lte_idx > -1:
            ct = "LESS_THAN_EQUAL"
            right_text = clause[lte_idx + 2 :]
        else:
            ct = "LESS_THAN"
            right_text = clause[lt_idx + 1 :]
    if gt_idx > 0:
        eq_idx = -1
        left_text = clause[:gt_idx]
        if gte_idx > 0:
            ct = "GREATER_THAN_EQUAL"
            right_text = clause[gte_idx + 2 :]
        else:
            ct = "GREATER_THAN"
            right_text = clause[gt_idx + 1 :]
    if eq_idx > -1:
        left_text = clause[:eq_idx]
        if clause[eq_idx + 1 : eq_idx + 3] == "(+-":
            eq = clause[eq_idx : clause.find(")", eq_idx) + 1]
            ct = {"close_tolerance": int(eq[4:-1])}
            right_text = clause[eq_idx + len(eq) + 1 :]
        elif clause[eq_idx + 1 : eq_idx + 3] == "#=":
            ct = "MEMID_EQUAL"
            right_text = clause[eq_idx + 3 :]
        else:
            ct = "EQUAL"
            right_text = clause[eq_idx + 1 :]
    if mod_idx > -1:
        # %_(modulus)(+-close_tolerance)
        left_text = clause[:mod_idx]
        mod_text = clause[eq_idx : clause.find(" ", mod_idx)]
        open_paren_idx = mod_text.find("(")
        close_paren_idx = mod_text.find(")")
        mod = int(mod_text[open_paren_idx + 1 : close_paren_idx])
        open_paren_idx = mod_text.find("(", close_paren_idx)
        if open_paren_idx > -1:
            close_paren_idx = mod_text.find(")", open_paren_idx)
            tol = int(mod_text[open_paren_idx + 3 : close_paren_idx])
            ct = {"close_tolerance": tol, "modulus": mod}
        else:
            ct = {"close_tolerance": tol, "modulus": mod}
        right_text = clause[eq_idx + len(mod_text) :]

    left_value = maybe_eval_literal(left_text.strip())
    # TODO warn if right_value was something that needed eval?
    right_value = maybe_eval_literal(right_text.strip())
    f = {
        "input_left": {"value_extractor": {"attribute": left_value}},
        "input_right": {"value_extractor": right_value},
        "comparison_type": ct,
    }

    return f


def convert_where_from_sqly(clause, d):
    tree = treeify_sqly_where(clause)
    if type(tree) is str:
        tree = {"AND": [tree]}
    d["where_clause"] = convert_where_tree(tree)


def convert_output_from_sqly(clause, d):
    # FIXME !!! deal with recursion.  what if there is sqly in attribute?
    # can be attribute, or list of simple attributes in form (a; b; c)
    # currently cannot handle a list with a complex (new filters style) attribute
    if clause == "MEMORY" or clause == "COUNT":
        output = clause
    else:
        pidx = match_symbol(clause)
        if pidx > -1:
            oidx = clause.find("(")
            clause = clause[oidx + 1 : pidx]
            # this WILL break for FILTERs style attributes
            if clause.find(",") > -1:
                attrs = [c.strip() for c in clause.split(",")]
            else:
                attrs = [clause]
            output = [{"attribute": maybe_eval_literal(a)} for a in attrs]
        else:
            output = [{"attribute": maybe_eval_literal(clause)}]

    d["output"] = output


def convert_memtype_from_sqly(clause, d):
    # FIXME allow sentences with OR
    d["memory_type"] = clause


def convert_order_by_from_sqly(clause, d):
    if not d.get("selector"):
        d["selector"] = {}
    l = clause.find("LOCATION")
    if l == 0:
        d["selector"]["location"] = maybe_eval_literal(clause[9:])
    else:
        if clause == "RANDOM":
            d["selector"]["return_quantity"] = "random"
        else:
            d["selector"]["return_quantity"] = {"argval": {"quantity": {}}}
            d["selector"]["return_quantity"]["argval"]["quantity"][
                "attribute"
            ] = maybe_eval_literal(clause)


def convert_limit_from_sqly(clause, d):
    # this requires doing  convert_order_by first
    c = clause.split()
    assert d.get("selector")
    d["selector"]["ordinal"] = int(c[0])
    if len(c) > 1:
        d["selector"]["return_quantity"]["argval"]["polarity"] = {"DESC": "MAX", "ASC": "MIN"}[
            c[1]
        ]


def convert_coref_from_sqly(clause, d):
    d["contains_coreference"] = clause


def convert_same_from_sqly(clause, d):
    if not d.get("selector"):
        d["selector"] = {}
    d["selector"]["same"] = clause


if __name__ == "__main__":
    from droidlet.interpreter.tests import all_test_commands

    has_name_cow = {
        "input_left": {"value_extractor": "has_name"},
        "input_right": {"value_extractor": "cow"},
        "comparison_type": "EQUAL",
    }
    has_colour_green = {
        "input_left": {"value_extractor": "has_colour"},
        "input_right": {"value_extractor": "green"},
        "comparison_type": "EQUAL",
    }
    has_colour_red = {
        "input_left": {"value_extractor": "has_colour"},
        "input_right": {"value_extractor": "red"},
        "comparison_type": "EQUAL",
    }

    distance_to_me_greater_5 = {
        "input_left": {"value_extractor": all_test_commands.ATTRIBUTES["distance from me"]},
        "input_right": {"value_extractor": 5},
        "comparison_type": "GREATER_THAN",
    }

    c = {"AND": [has_name_cow, {"OR": [has_colour_green, has_colour_red]}]}
    x = sqlyify_where_clause(c)
    print(x)

    #    d = {"AND": [distance_to_me_greater_5, has_name_cow, {"OR": [has_colour_green, has_colour_red]}]}
    d = {"AND": [has_colour_green, has_name_cow, {"OR": [has_colour_green, has_colour_red]}]}

    z = sqlyify_where_clause(d)
    print(z)

    f = all_test_commands.FILTERS["that cow"]
    s = new_filters_to_sqly(f)

    S = {}
    F = {}
    FF = {}

    for k, f in all_test_commands.FILTERS.items():
        S[k] = f
        F[k] = new_filters_to_sqly(S[k])
        FF[k] = sqly_to_new_filters(F[k])
