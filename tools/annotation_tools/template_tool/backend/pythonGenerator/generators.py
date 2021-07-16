""""
Copyright (c) Facebook, Inc. and its affiliates.

This file defines a Generator class to generate surface form,
logical form pairs for templates saved using the template tool interface
"""

import random
import json
from nested_lookup import nested_update
from nested_lookup import nested_lookup
from deepmerge import always_merger
from pprint import pprint
import copy
import xml.etree.ElementTree as ET


def getSpanKeys(d):
    if d is None:
        return
    for k, v in d.items():
        if v == "":
            yield k
        elif type(v) == dict:
            yield from getSpanKeys(v)


def get_absolute_path_to_key(json, key):
    if not isinstance(json, dict):
        return None
    if key in json.keys():
        return key
    ans = None
    for json_key in json.keys():
        r = get_absolute_path_to_key(json[json_key], key)
        if r is None:
            continue
        else:
            ans = "{}.{}".format(json_key, r)
    return ans


def set_span(code, surface_form, span_value):
    """This function sets the span value in a dictionary given
    a span value"""
    span_array = span_value.split(" ")
    surface_form_words = surface_form.split(" ")
    start_span = surface_form_words.index(span_array[0])
    end_span = start_span + len(span_array) - 1
    span = [0, [start_span, end_span]]
    spanKeys = getSpanKeys(copy.deepcopy(code))
    triple_position = ""
    triples_data = []
    span_keys = []
    for spans in spanKeys:
        triples_data.append({"pred_text": spans, "obj_text": span})
        triple_position = ".".join(get_absolute_path_to_key(code, spans).split(".")[:-1])
        span_keys.append(spans)
    # handle triples here: Add a key called 'triples' and delete the
    # previous 'has_x' key
    if triples_data:
        k_code = code
        for k in triple_position.split("."):
            k_code = k_code[k]
        for key in span_keys:
            k_code.pop(key)
        k_code["triples"] = triples_data
    return code


class Generator:
    """This is a Generator class that is initialised for templates
    using template information saved from the template interface"""

    def __init__(self, information):

        # change this to change the number of generations
        self.n = 1
        self.num = 0

        # information about the template
        self.info = information

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __generate__(self):
        # generation = None
        # while self.num < self.n:
        #     generations.append(self.next())
        #     self.num += 1
        # print(generations[0])
        generation = self.next()
        return generation

    def next(self):
        """next function following the iterable pattern"""
        cur = self.get_generation()
        return cur

    def get_generation(self):
        """This function returns the generations for a template"""
        return generate_template(self.info)


def generate_template(info):
    """This function generates template surface-logical
    forms given information about the template"""
    surface_form = ""
    chosen_surface_forms = []
    for target_list in info["surfaceForms"]:
        if target_list:
            choose_surface_form = random.choice(target_list)
            surface_form += choose_surface_form + " "
            chosen_surface_forms.append(choose_surface_form)
        else:
            # no surface form associated with the template object
            chosen_surface_forms.append("")
    try:
        for i in range(len(info["code"])):
            cur_code = info["code"][i]
            try:
                span_value = spans[chosen_surface_forms[i]]
            except BaseException:
                # no span value
                span_value = chosen_surface_forms[i]
            # set the span
            info["code"][i] = set_span(cur_code, surface_form, span_value)
    except BaseException:
        # no logical form associated with the template object
        info["code"] = {}
    dictionary = {}
    dictionary = generate_dictionary(info["code"])
    surface_form = surface_form.strip()
    return [surface_form, dictionary]


def generate_dictionary(code, i=0, skeletal=None):
    """This function generates the action dictionary given an array
    of action dictionaries"""
    if skeletal is None:
        skeletal = {}
    if i == len(code):
        # all action dictionaries have been merged
        return skeletal
    found = False
    if code[i]:
        cur_code = code[i]
        key = list(cur_code.keys())[0]
        if nested_lookup(key, skeletal):
            # the parent key exists
            found = True
            cur_value = nested_lookup(key, skeletal)[0]
            new_value = always_merger.merge(cur_value, cur_code[key])
            nested_update(skeletal, key, new_value)
        if not found:
            skeletal = always_merger.merge(skeletal, cur_code)
    return generate_dictionary(code, i=i + 1, skeletal=skeletal)


def getBLockType(savedBlocks, block_name):
    treeOne = ET.fromstring(savedBlocks[block_name])
    block_type = treeOne[0].attrib["type"]
    return block_type


def update_list_value(d, rnd_index):
    new_d = copy.deepcopy(d)
    for k, v in d.items():
        if type(v) == list:
            new_d[k] = v[rnd_index]
        elif type(v) == dict:
            new_d[k] = update_list_value(v, rnd_index)
        else:
            new_d[k] = v
    return new_d


def fixTemplatesWithRandomBlock(codeList, surfaceFormList):
    updatedCodeList, updatedSurfaceFormList = [], []
    for code, surfaceForm in zip(codeList, surfaceFormList):
        if surfaceForm and type(surfaceForm[0]) == list:
            rnd_index = random.choice(range(len(surfaceForm)))
            if code is None:
                updatedCodeList.append(code)
            # code is list
            if type(code) == list:
                updatedCodeList.append(code[rnd_index])
            else:
                # code has a nested list somewhere in the dict
                updatedCode = update_list_value(code, rnd_index)
                updatedCodeList.append(updatedCode)
            updatedSurfaceFormList.append(surfaceForm[rnd_index])
        else:
            updatedCodeList.append(code)
            updatedSurfaceFormList.append(surfaceForm)
    return updatedCodeList, updatedSurfaceFormList


def getAllTemplates(template_data):
    spans = {}
    if "spans" in template_data:
        spans = template_data["spans"]
    templatesSaved = template_data["templates"]
    savedBlocks = template_data["savedBlocks"]
    templates = {}

    for k, v in templatesSaved.items():
        templateContent = v
        templateContentCopy = copy.deepcopy(templateContent)

        # fix random blocks to have one code block and one surface form
        if getBLockType(savedBlocks, k) == "random":
            rnd_index = random.choice(range(len(templateContent["code"])))
            code = templateContent["code"][rnd_index]
            surfaceForm = templateContent["surfaceForms"][rnd_index]
            templateContentCopy["code"] = code
            templateContentCopy["surfaceForm"] = surfaceForm
        info = {}
        info["code"] = {}

        if "code" in templateContentCopy.keys():
            if not isinstance(templateContentCopy["code"], list):
                # skip template objects...
                # info['code'] = [info['code']]
                continue
            info["code"] = templateContentCopy["code"]
        # Template object with no code
        if not info["code"]:
            continue

        info["spans"] = spans
        info["surfaceForms"] = templateContentCopy["surfaceForms"]

        # if not isinstance(info['surfaceForms'][0], list):
        #     # it is a surface form
        #     info['surfaceForms'] = [info['surfaceForms']]

        code, surfaceForm = fixTemplatesWithRandomBlock(info["code"], info["surfaceForms"])
        info["code"] = code
        info["surfaceForms"] = surfaceForm
        templates[k] = info
    return templates


def generatePairs(templates, num_gens):
    # initialise an array of generators
    arrayOfObjects = []
    # generate num_gens pairs
    for i in range(num_gens):
        # pick a random template
        templateName = random.choice(list(templates))
        info = copy.deepcopy(templates[templateName])
        arrayOfObjects.append([templateName, Generator(info)])

    return arrayOfObjects


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gens", type=int, default=100)
    parser.add_argument("--out_file", type=str, default="out.txt")
    parser.add_argument("--input_file", type=str, default="./../templates.txt")
    parser.add_argument(
        "--format",
        action="store_true",
        help="this flag tells the script to generate formatted training data",
    )
    args = parser.parse_args()

    template_data = {}

    with open(args.input_file, "r") as f:
        template_data = json.load(f)

    templates = getAllTemplates(template_data)
    if "null" in templates:
        templates.pop("null")
    generationPairs = generatePairs(templates, args.num_gens)

    with open(args.out_file, "w") as f:
        for obj in generationPairs:
            template_name, generation_obj = obj
            # generate logical-surface form pair array for the template
            text, action_dict = generation_obj.__generate__()
            updated_dict = action_dict
            # it is of type HUMAN_GIVE_COMMAND
            if "dialogue_type" not in action_dict:
                updated_dict = {
                    "dialogue_type": "HUMAN_GIVE_COMMAND",
                    "action_sequence": [action_dict],
                }
            if args.format:
                f.write(text + "|" + json.dumps(updated_dict) + "\n")
            else:
                print(text)
                pprint(updated_dict)
                print()
