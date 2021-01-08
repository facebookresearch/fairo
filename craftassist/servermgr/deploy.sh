#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.



cd $(dirname $0)/../../
git push heroku_servermgr $(git subtree split --prefix python/servermgr):master --force
