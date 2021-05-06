#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



cd $(dirname $0)
coverage run --source . -m unittest discover
status=$?
coverage report
coverage xml -o /shared/test_MC.xml
exit $status
