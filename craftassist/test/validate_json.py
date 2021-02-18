from jsonschema import validate, exceptions, RefResolver, Draft7Validator
import json
import ast
from pprint import pprint
import os
import argparse
import glob
import re


class JSONValidator:
    def __init__(self, schema_dir, use_span_values):
        """Initialize a JSON validator by loading all schemas and resolving references.

        Args:
        schema_dir (str) -- Where to look for subschemas.
        use_span_values (bool) -- Whether to use the value of the span or the array default in the spec.
        """
        # RefResolver initialization requires a base schema and URI
        base_uri = args.schema_dir + "grammar_spec.schema.json"
        base_schema = json.load(open(base_uri))
        # NOTE: Though not required, naming convention is that schemas end in .schema.json
        re_pattern = "(.*)\/(.*).schema.json$"
        base_schema_name = re.search(re_pattern, base_uri).group(2)
        resolver = RefResolver(base_schema_name + ".schema.json", base_schema)
        
        schemas_to_exclude = []
        # Which type of span to load
        if args.use_span_values:
            span_schema_name = "string_span"
            schemas_to_exclude += "array_span"
        else:
            span_schema_name = "array_span"
            schemas_to_exclude += "string_span"
        # Load all subschemas in schema directory
        for schema_path in glob.glob(args.schema_dir + "*.json"):
            schema_name = re.search(re_pattern, schema_path).group(2)
            print(schema_name)
            json_schema = json.load(open(schema_path))
            if schema_name in schemas_to_exclude:
                continue
            elif span_schema_name in schema_path:
                resolver.store["span" + ".schema.json"] = json_schema
            else:
                resolver.store[schema_name + ".schema.json"] = json_schema
        self.base_schema = base_schema
        self.resolver = resolver

    def validate_data(self, data_path):
        """
        Validates JSON style parse trees from a dataset, where each row has the format
        [command] | [parse_tree]

        Args:
        data_path (str) -- Path to plaintext file, eg. annotated.txt
        schema (jsonschema) -- JSON schema we are validating against
        resolver (RefResolver) -- A store of subschemas referenced in the main action dict spec.
        """
        with open(data_path) as fd:
            dataset = fd.readlines()
            for line in dataset:
                command, action_dict = line.split("|")
                parse_tree = ast.literal_eval(action_dict)
                try:
                    validate(instance=parse_tree, schema=self.base_schema, resolver=self.resolver)
                except exceptions.ValidationError as e:
                    print(command)
                    pprint(parse_tree)
                    print("\n")

    def validate_instance(self, parse_tree, schema, resolver):
        """
        Validates a parse tree instance.

        Args:
        parse_tree (dict) -- dictionary we want to validate
        schema (jsonschema) -- JSON schema we are validating against
        resolver (RefResolver) -- A store of subschemas referenced in the main action dict spec.
        """
        try:
            validate(instance=parse_tree, schema=schema, resolver=resolver)
        except exceptions.ValidationError as e:
            print("Error validating:\n{}\n".format(parse_tree))
            print(e)
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
    parser.add_argument(
        "--use_span_values",
        default=False,
        action="store_true",
        help="whether to use span values instead of the array",
    )
    args = parser.parse_args()
    json_validator = JSONValidator(args.schema_dir, args.use_span_values)

    # Validate dataset against schema using resolver to resolve cross references
    json_validator.validate_data(args.data_path)