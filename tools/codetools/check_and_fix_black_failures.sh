#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

set -e
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/../..

echo '==== Black ====='

CHECK_FILES="base_agent craftassist droidlet"

black $CHECK_FILES

if [[ $1 == "--ci" ]]; then
  if [ -n "$(git status --porcelain)" ]; then
    git config user.name >/dev/null || git config --global user.name "Yuxuan Sun"
    git config user.email >/dev/null || git config --global user.email "yuxuans@fb.com"
    git add $CHECK_FILES && git commit -m "[skip ci] Automatic style fix" && git push --set-upstream origin $(git rev-parse --abbrev-ref HEAD)
    echo "Auto fix style."
  else
    echo "Style is perfect. No need to fix."
  fi
fi

echo
echo
echo
