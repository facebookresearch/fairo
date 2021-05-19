"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""This file has functions that an help edit the config
got craftassist_cuberite."""
import config_parser


def set_seed(path, seed):
    with config_parser.inplace_edit(path) as config:
        config["Seed"]["Seed"] = seed


def add_plugin(path, plugin):
    with config_parser.inplace_edit(path) as config:
        config["Plugins"]["Plugin"].append(plugin)


def remove_plugin(path, plugin):
    with config_parser.inplace_edit(path) as config:
        config["Plugins"]["Plugin"].remove(plugin)


def set_mode_survival(path):
    with config_parser.inplace_edit(path) as config:
        config["General"]["GameMode"] = 0


def set_mode_creative(path):
    with config_parser.inplace_edit(path) as config:
        config["General"]["GameMode"] = 1


def set_port(path, port):
    with config_parser.inplace_edit(path) as config:
        config["Server"]["Ports"] = port
