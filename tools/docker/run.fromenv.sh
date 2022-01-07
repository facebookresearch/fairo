#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


set -u

echo $RUN_SH_GZ_B64 | base64 --decode | gunzip > /run.sh
chmod +x /run.sh
. /run.sh
