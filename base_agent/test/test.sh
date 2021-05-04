#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


cd $(dirname $0)
coverage run --source . -m unittest discover -s . -t ./
status=$?
coverage report
coverage xml -o /shared/test_base_agent.xml
exit $status
