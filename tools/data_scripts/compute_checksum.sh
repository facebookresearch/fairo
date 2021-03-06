#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script computes hashes for local directories and saves them to data_scripts/default_checksums/
# ./compute_and_upload_checksum.sh craftassist # hash for semantic parser models
# ./compute_and_upload_checksum.sh craftassist datasets # hash for datasets folder
# ./compute_and_upload_checksum.sh locobot # hash for locobot models

function pyabspath() {
    python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" $1
}

ROOTDIR=$(pyabspath $(dirname "$0")/../../)
echo "Rootdir $ROOTDIR"

if [ -z $1 ]
then
    AGENT="craftassist"
    echo "Agent name not specified, defaulting to craftassist"
else
	AGENT=$1
fi

if [ -z $2 ]
then
    HASH_PATH="models"
    echo "Path not specified, defaulting to models/"
else
	HASH_PATH=$2
fi

echo $HASH_PATH
cd ${ROOTDIR}/$AGENT/agent/

# saves sha1sum for CHECKSUM_FOLDER at SAVE_TO_PATH 
save_checksum() {
    SAVE_TO_PATH=$1
    CHECKSUM_FOLDER=$2
    echo "Before " $(cat $SAVE_TO_PATH)
    find $CHECKSUM_FOLDER -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum | tr -d '-' | xargs > $SAVE_TO_PATH
    echo "After " $(cat $SAVE_TO_PATH)
}

echo "Computing hashes ..."
if [ "$HASH_PATH" = "models" ]
then
    if [ $AGENT == "locobot" ]; then
        save_checksum "${ROOTDIR}/tools/data_scripts/default_checksums/locobot.txt" "models/perception" 
    else # craftassist
        save_checksum "${ROOTDIR}/tools/data_scripts/default_checksums/nsp.txt" "models/semantic_parser"
    fi
else # datasets
    save_checksum "${ROOTDIR}/tools/data_scripts/default_checksums/datasets.txt" $HASH_PATH/ 
fi

