#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



cd $(dirname $0)

SHARED_PATH=/shared
COV_RELATIVE=craftassist

pytest --cov-report=xml:$SHARED_PATH/test_MC.xml --cov=$COV_RELATIVE . --disable-pytest-warnings
status=$?
exit $status
