from copy import deepcopy
import ast

FILTERS_KW = ["SELECT", "FROM", "WHERE", "ORDER BY", "LIMIT", "SAME", "CONTAINS_COREFERENCE"]
LIMITS = {"FIRST": "1", "SECOND": "2", "THIRD": "3"}

# name resolution for properties:  DOES NOT EXIST
# that is: "triples"/"properties"/"column names" are equivalent, and a
# single memory is allowed to have multiple values for a single property!!!
# this sounds weird when you think about many standard table columns, but
# for others its natural, and certainly for Triples its natural...


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


def convert_triple_to_comparator(triple):
    # triples in FILTERS rn are only being interpreted when they are searches over subj.
    # with a fixed predicate.
    pred_text = triple.get("pred_text")
    if not pred_text:
        raise Exception("triples currently need a pred_text in FILTERS form")
    obj = triple.get("obj_text") or triple.get("obj")
    if not obj:
        raise Exception("triples currently need a obj_text or obj in FILTERS form")
    # this is post span/coref resolve
    c = {
        "input_left": {"value_extractor": pred_text},
        "input_right": {"value_extractor": obj},
        "comparison_type": "EQUAL",
    }
    return c


def old_filters_to_new_filters(d):
    where_clause = []
    new_filters = deepcopy(d)
    if d.get("triples"):
        del new_filters["triples"]
        for triple in d["triples"]:
            c = convert_triple_to_comparator(triple)
            where_clause.append(c)
    if d.get("comparator"):
        del new_filters["comparator"]
        for c in d["comparator"]:
            where_clause.append(c)
    if d.get("author"):
        del new_filters["author"]
        c = (
            {
                "input_left": {"value_extractor": "author"},
                "input_right": {"value_extractor": d["author"]},
                "comparison_type": "EQUAL",
            },
        )
        where_clause.append(c)
    if where_clause:
        new_filters["where_clause"] = {"AND": where_clause}
    # FIXME: moving ordinal in to root of selector dict
    if d.get("selector"):
        s = d["selector"]
        if s.get("return_quantity", {}).get("argval", {}).get("ordinal"):
            o = deepcopy(s["return_quantity"]["argval"]["ordinal"])
            # ordinal at root of selector:
            new_filters["selector"]["ordinal"] = o
            del new_filters["selector"]["return_quantity"]["argval"]["ordinal"]
        elif s.get("return_quantity", {}).get("random"):
            new_filters["selector"]["ordinal"] = deepcopy(s["return_quantity"]["random"])
            new_filters["selector"]["return_quantity"] = "random"

    return new_filters


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
                input_left = str(clause["input_left"]["value_extractor"])
                input_right = str(clause["input_right"]["value_extractor"])
                inequality_symbol = get_inequality_symbol(clause["comparison_type"])
                s = input_left + " " + inequality_symbol + " " + input_right
                if clause.get("comparison_measure"):
                    s = s + " MEASURED_IN " + clause["comparison_measure"] + " "
            else:
                s = sqlyify_where_clause(clause)
            clause_texts.append(s)
        if k == "NOT":
            assert len(clause_texts) == 1
            return "( NOT " + clause_texts[0] + " ) "
        else:
            return "(" + (" " + k + " ").join(clause_texts) + ")"


def new_filters_to_sqly(d):
    S = "SELECT "
    o = d.get("output", "MEMORY")
    if o == "MEMORY" or o == "COUNT":
        S = S + o + " "
    else:
        if o.get("attribute") is None:
            raise Exception("malformed output dict {}".format(o))
        a = str(o["attribute"])
        # FIXME do this more reliably
        if (
            a[0] == "{"
            or len(a.split()) > 1
            or any([k in a for k in ["SELECT", "WHERE", "ORDER BY"]])
        ):
            a = "(" + a + ")"
        S = S + a + " "
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


def close_paren(S, pidx=0):
    """
    find the paren closing the first open paren after pidx
    """
    pidx = S.find("(", pidx)
    if pidx < 0:
        return pidx
    count = 1
    while count > 0:
        o = S.find("(", pidx + 1)
        c = S.find(")", pidx + 1)
        if c > 0:
            if c < o or o < 0:
                count = count - 1
                pidx = c
            else:
                count = count + 1
                pidx = o
        else:
            # parens not balanced
            return -1
    return pidx


def find_keyword(S, start=0, keywords=FILTERS_KW):
    # find the first keyword present in the string
    for kw in keywords:
        kidx = S.find(kw, start)
        if kidx >= 0:
            break
    return kidx


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
        pidx = close_paren(S)
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
    p = close_paren(S)
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


def split_where(clause):
    if clause[0] == "(":
        c = close_paren(clause)
        if len(clause) == c + 1:
            clause = clause[1:-1]
    return split_sqly(clause, keywords=["AND", "OR"])


def treeify_sqly_where(clause):
    """ 
    converts a where clause in sqly form to a nested dict:
    for example:
    (has_name = cow AND (has_colour = green OR has_colour = red))
    --> 
    {'AND': ['has_name = cow', {'OR': ['has_colour = green', 'has_colour = red']}]}
    """
    t = split_where(clause)
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
        left_text = clause[:lt_idx]
        if lte_idx > -1:
            ct = "LESS_THAN_EQUAL"
            right_text = clause[lte_idx + 2 :]
        else:
            ct = "LESS_THAN"
            right_text = clause[lt_idx + 1 :]
    if gt_idx > 0:
        left_text = clause[:gt_idx]
        if gte_idx > 0:
            ct = "GREATER_THAN_EQUAL"
            right_text = clause[gte_idx + 2 :]
        else:
            ct = "GREATER_THAN"
            right_text = clause[gte_idx + 1 :]
    if eq_idx > -1:
        left_text = clause[:eq_idx]
        if clause[eq_idx + 1 : eq_idx + 3] == "(+-":
            eq = clause[eq_idx : clause.find(")", eq_idx) + 1]
            ct = {"close_tolerance": int(eq[4:-1])}
            right_text = clause[eq_idx + len(eq) + 1 :]
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

    left_value = maybe_eval_literal(left_text)
    right_value = maybe_eval_literal(right_text)
    f = {
        "input_left": {"value_extractor": left_value},
        "input_right": {"value_extractor": right_value},
        "comparison_type": ct,
    }

    return f


def maybe_eval_literal(clause):
    try:
        output = ast.literal_eval(clause)
    except:
        output = clause
    if type(output) is tuple:
        output = output[0]
    return output


def convert_output_from_sqly(clause, d):
    # FIXME !!! deal with recursion.  what if there is sqly in attribute?
    # can be attribute or
    if clause == "MEMORY" or clause == "COUNT":
        output = clause
    else:
        output = {"attribute": maybe_eval_literal(clause)}
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


def convert_where_from_sqly(clause, d):
    tree = treeify_sqly_where(clause)
    if type(tree) is str:
        tree = {"AND": [tree]}
    d["where_clause"] = convert_where_tree(tree)


def convert_coref_from_sqly(clause, d):
    d["contains_coreference"] = clause


def convert_same_from_sqly(clause, d):
    if not d.get("selector"):
        d["selector"] = {}
    d["selector"]["same"] = clause


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

    f = old_filters_to_new_filters(all_test_commands.FILTERS["that cow"])
    s = new_filters_to_sqly(f)

    S = {}
    F = {}
    FF = {}

    for k, f in all_test_commands.FILTERS.items():
        S[k] = old_filters_to_new_filters(f)
        F[k] = new_filters_to_sqly(S[k])
        FF[k] = sqly_to_new_filters(F[k])
