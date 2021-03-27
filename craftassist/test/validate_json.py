from jsonschema import validate, exceptions, RefResolver, Draft7Validator
import json
import ast
from pprint import pprint
import os
import argparse
import glob
import re


class JSONValidator:
    def __init__(self, schema_dir, span_type):
        """Initialize a JSON validator by loading all schemas and resolving references.

        Args:
        schema_dir (str) -- Where to look for subschemas.
        span_type (str) -- Span types to allow, eg. array or string.
        """
        # RefResolver initialization requires a base schema and URI
        base_uri = schema_dir + "grammar_spec.schema.json"
        try:
            with open(base_uri) as fd:
                base_schema = json.load(fd)
        except Exception as e:
            print(e)
            raise e
        # NOTE: Though not required, naming convention is that schemas end in .schema.json
        re_pattern = "(.*)\/(.*).schema.json$"
        base_schema_name = re.search(re_pattern, base_uri).group(2)
        resolver = RefResolver(base_schema_name + ".schema.json", base_schema)
        
        span_schemas = ["string_span", "array_span", "array_and_string_span"]
        # Which type of span to load
        if span_type == "string":
            span_schema_name = "string_span"
        elif span_type == "array":
            span_schema_name = "array_span"
        elif span_type == "all":
            span_schema_name = "array_and_string_span"
        else:
            print("I don't recognize span type {}.".format(span_type))
            raise Exception

        # Load all subschemas in schema directory
        for schema_path in glob.glob(schema_dir + "*.json"):
            schema_name = re.search(re_pattern, schema_path).group(2)
            with open(schema_path) as fd:
                json_schema = json.load(fd)
            if schema_name == span_schema_name:
                resolver.store["span" + ".schema.json"] = json_schema
            elif schema_name in span_schemas:
                continue
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
                parse_tree = json.loads(action_dict)
                try:
                    validate(instance=parse_tree, schema=self.base_schema, resolver=self.resolver)
                except exceptions.ValidationError as e:
                    print(command)
                    pprint(parse_tree)
                    print(e)
                    print("\n")

    def validate_instance(self, parse_tree):
        """
        Validates a parse tree instance.

        Args:
        parse_tree (dict) -- dictionary we want to validate
        """
        try:
            validate(instance=parse_tree, schema=self.base_schema, resolver=self.resolver)
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
        "--span_type",
        type=str,
        default="array",
        choices=["string", "array", "all"],
        help="What span types to allow",
    )
    args = parser.parse_args()
    json_validator = JSONValidator(args.schema_dir, args.span_type)

    # Validate dataset against schema using resolver to resolve cross references
    json_validator.validate_data(args.data_path)