from jsonschema import validate, exceptions, RefResolver, Draft7Validator
import json
import ast
from pprint import pprint
import os
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="craftassist/agent/datasets/full_data/templated_filters.txt",
        help="path to dataset with examples we want to validate, where each row contains a command and parse tree separated by |.",
    )
    args = parser.parse_args()
    schema = json.load(open("grammar_spec.schema.json"))
    filters_schema = json.load(open("filters.schema.json"))
    action_dict_components = json.load(open("action_dict_components.schema.json"))
    other_dialogue = json.load(open("other_dialogue.schema.json"))

    resolver = RefResolver("filters.schema.json", filters_schema)
    resolver.store["grammar_spec.schema.json"] = schema
    resolver.store["action_dict_components.schema.json"] = action_dict_components
    resolver.store["other_dialogue.schema.json"] = other_dialogue
    validate_data(args.data_path, schema, resolver)