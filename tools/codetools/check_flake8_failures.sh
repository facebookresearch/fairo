#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# part of it taken from ParlAI

set -e
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

reroot() {
    # possibly rewrite all filenames if root is nonempty
    if [[ "$1" != "" ]]; then
        cat | xargs -I '{}' realpath --relative-to=. $1/'{}'
    else
        cat
    fi
}

onlyexists() {
    # filter filenames based on what exists on disk
    while read fn; do
        if [ -f "${fn}" ]; then
            echo "$fn"
        fi
    done
}

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/../..

# check flake8 installation
command -v flake8 >/dev/null || \
    ( echo "Please run \`pip install flake8\` and rerun $0." && false )

# parse args
RUN_ALL_FILES=0

if [[ $1 == "--ci" ]]; then
    # find out what files we're working on
    if [[ $RUN_ALL_FILES -eq 1 ]]; then
        CHECK_FILES="$(git $REPO ls-files | grep '\.py$' | reroot $ROOT | onlyexists $ROOT | tr '\n' ' ')"
    else
        CHECK_FILES="$(git $REPO diff --name-only master... | grep '\.py$' | reroot $ROOT | onlyexists | tr '\n' ' ')"
    fi
    if [[ $CHECK_FILES != "" ]]; then
        # run flake8 only on working files
        flake8 $CHECK_FILES
    fi
else
    # run flake8 locally on all files
    flake8 ./base_agent/
fi