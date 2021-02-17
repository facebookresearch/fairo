from jsonschema import validate, exceptions, RefResolver, Draft7Validator
import json
import ast
from pprint import pprint
import os
import argparse
import glob
import re

"""
Validates JSON style parse trees from a dataset, where each row has the format
[command] | [parse_tree]

Args:
data_path (str) -- Path to plaintext file, eg. annotated.txt
schema (jsonschema) -- JSON schema we are validating against
resolver (RefResolver) -- A store of subschemas referenced in the main action dict spec.
"""
def validate_data(data_path, schema, resolver):
    with open(data_path) as fd:
        dataset = fd.readlines()
        for line in dataset:
            command, action_dict = line.split("|")
            parse_tree = ast.literal_eval(action_dict)
            try:
                validate(instance=parse_tree, schema=schema, resolver=resolver)
            except exceptions.ValidationError as e:
                print(command)
                pprint(parse_tree)
                print("\n")


def validate_instance(parse_tree, schema, resolver):
    try:
        validate(instance=parse_tree, schema=schema, resolver=resolver)
    except exceptions.ValidationError as e:
        print("Error validating:\n{}\n".format(parse_tree))
        return False
    return True                          


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="craftassist/agent/datasets/full_data/templated_filters.txt",
        help="path to dataset with examples we want to validate, where each row contains a command and parse tree separated by |.",
    )
    parser.add_argument(
        "--schema_dir",
        type=str,
        default="base_agent/documents/json_schema/",
        help="path to directory containing JSON schemas we want to load",
    )
    args = parser.parse_args()
    # RefResolver initialization requires a base schema and URI
    base_uri = args.schema_dir + "grammar_spec.schema.json"
    base_schema = json.load(open(base_uri))
    # NOTE: Though not required, naming convention is that schemas end in .schema.json
    re_pattern = "(.*)\/(.*).schema.json$"
    base_schema_name = re.search(re_pattern, base_uri).group(2)
    resolver = RefResolver(base_schema_name + ".schema.json", base_schema)
    # Load all subschemas in schema directory
    for schema_path in glob.glob(args.schema_dir + "*.json"):
        schema_name = re.search(re_pattern, schema_path).group(2)
        json_schema = json.load(open(schema_path))
        resolver.store[schema_name + ".schema.json"] = json_schema
    # Validate dataset against schema using resolver to resolve cross references
    validate_data(args.data_path, base_schema, resolver)