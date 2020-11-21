#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.


$(aws ecr get-login | sed 's/-e none//')
