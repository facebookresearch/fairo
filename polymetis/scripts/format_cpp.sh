#!/bin/bash -e

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e 
set -o pipefail

if [[ "$1" != "check" ]] && [[ "$1" != "format" ]]; then
  echo "Type 'check' to verify formatting, and 'format' to format recursively."
  echo "By default, only operates on staged files in git."
  echo "Add 'all' to the command (e.g. 'check all') to go through all source files."
  exit 1
fi

if [[ "$2" == "all" ]]; then
  SOURCE_FILES=`find . -name \*.cpp -type f -or -name \*.h -type f -or -name \*.hpp -type f | grep -v third_party`
else
  SOURCE_FILES=`(git diff --name-only --cached | grep '.*\.cpp$\|.*\.h$\|.*\.hpp$') || :`
  if [[ "$SOURCE_FILES" == "" ]]; then
    exit 0
  fi
fi

# Modified from https://github.com/status-im/react-native-desktop/blob/ee98cce6ca133ca016a389dd83258968dbea5736/.circleci/config.yml

if [[ "$1" == "format" ]]; then
  echo $SOURCE_FILES | xargs clang-format -i --verbose
elif [[ "$1" == "check" ]]; then
  for SOURCE_FILE in $SOURCE_FILES
  do
    export FORMATTING_ISSUE_COUNT=`clang-format -output-replacements-xml $SOURCE_FILE | grep offset | wc -l`
    if [[ "$FORMATTING_ISSUE_COUNT" -gt "0" ]]; then
      echo "Source file $SOURCE_FILE contains formatting issues. Please use clang-format tool to resolve found issues."
      exit 1
    fi
  done
else
  echo "Unknown argument $1"
  exit 1
fi
