"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import os

repo_home = os.path.dirname(os.path.realpath(__file__))


def path(p):
    return os.path.join(repo_home, p)
