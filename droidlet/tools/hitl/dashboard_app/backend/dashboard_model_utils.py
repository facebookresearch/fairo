"""
Copyright (c) Facebook, Inc. and its affiliates.

Utils for loading a model and preparing model infomation for transporting via socket.
"""
import argparse
import collections
import json
import torch


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
