#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

set -e
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/../..

echo '==== Black ====='

CHECK_DIRECTORIES="droidlet polymetis"
for CHECK_DIR in $CHECK_DIRECTORIES
do
  black $CHECK_DIR

  if [[ $1 == "--ci" ]]; then
    if [ -n "$(git status --porcelain)" ]; then
      git config user.name >/dev/null || git config --global user.name "bot"
      git config user.email >/dev/null || git config --global user.email "bot@fb.com"
      git add $CHECK_DIR && git commit -m "Automatic style fix for $CHECK_DIR" && git push --set-upstream origin $(git rev-parse --abbrev-ref HEAD)
      echo "Auto fix style."
    else
      echo "Style is perfect. No need to fix."
    fi
  fi
done

echo
echo
echo
