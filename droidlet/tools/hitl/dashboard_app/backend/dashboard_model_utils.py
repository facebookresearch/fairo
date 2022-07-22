"""
Copyright (c) Facebook, Inc. and its affiliates.

Utils for loading a model and preparing model infomation for transporting via socket.
"""
import argparse
import collections
import os
import torch
import json

ROOTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../")


def load_model(model_fpath: str):
    """
    generates a model instance from the model file
    """
    return torch.load(model_fpath)


def get_keys(model):
    """
    helper method to get all keys for the model
    """
    return list(model.keys())


def get_value_by_key(model, key):
    """
    helper method for getting a value in the model dict
    """
    value = model[key]

    # logic for preparing json
    if isinstance(value, argparse.Namespace) or isinstance(value, collections.OrderedDict):
        # for args and state_dict, need to dump the __dict__ field
        return json.dumps(value.__dict__)
    else:
        # otherwise, dump the whole value object
        return json.dumps(value)


def get_complete_model(model):
    """
    helper method to get the complete model
    """
    model_dict = {}
    # get all fields
    for key in model.keys():
        model_dict[key] = get_value_by_key(model, key)
    return json.dumps(model_dict)


def get_model_checksum_by_name_n_agent(model_name, agent=None):
    """
    helper method to get model checksum
    """
    checksum_name = ""
    if model_name == "nlu":
        checksum_name = "nlu.txt"
    elif model_name == "perception":
        if agent == "locobot":
            checksum_name = "locobot_perception.txt"
        elif agent == "craftassist":
            checksum_name = "craftassist_perception.txt"

    checksum_path = os.path.join(
        ROOTDIR, "droidlet/tools/artifact_scripts/tracked_checksums/" + checksum_name
    )

    if os.path.exists(checksum_path):
        f = open(checksum_path)
        checksum = f.read()
        f.close()
        return checksum, None
    else:
        return f"Cannot find checksum for model = {model_name}, agent = {agent}", 404
