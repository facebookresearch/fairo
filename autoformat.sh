#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# exit on errors
set -e

BLACK_SCRIPT_PATH="tools/codetools/check_and_fix_black_failures.sh"
FLAKE8_SCRIPT_PATH="tools/codetools/check_flake8_failures.sh"

sh "$BLACK_SCRIPT_PATH"
echo "Done!"
echo

echo "===Checking flake8 errors now, please manually fix those==="
sh "$FLAKE8_SCRIPT_PATH"
echo
echo "Done!"
echo
