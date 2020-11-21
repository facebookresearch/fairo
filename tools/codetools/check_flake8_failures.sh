#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



set -e
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/../..


echo '==== Checking Flake8 ===='

flake8 ./base_agent/
