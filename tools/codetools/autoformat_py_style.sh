#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/../..

black base_agent craftassist droidlet
