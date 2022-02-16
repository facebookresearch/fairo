"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

def parse_sql(query):
    query = query.split()
    table = ""
    operation = ""
    for word in query:
        if word.isupper():
            operation += word + " "
        else:
            table += word
    return table, operation

def format_query(query, *args):
    """Turns query and arguments into a structured format

    Args:
        query (string): The query to be run against the database

    Returns:
        dict: An ordered format of query keys and corresponding arguments
    """
    query_args = {}
    start_idx = query.find("(")
    end_idx = query.find(")")
    if start_idx != -1 and end_idx != -1:
        keys = query[start_idx + 1: end_idx]
        keys = keys.split(", ")
        query_args = dict(zip(keys, list(args)))
    return query_args