#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script computes hashes for local directories and uploads them to AWS. Used to store hashes for the deployed models.
# ./compute_and_upload_checksum.sh craftassist # uploads hash for semantic parser models
# ./compute_and_upload_checksum.sh craftassist datasets # uploads hash for datasets folder
# ./compute_and_upload_checksum.sh locobot # uploads hash for locobot models

ROOTDIR=$(readlink -f $(dirname "$0")/../../)
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

upload_path_to_s3() {
    CHECKSUM_PATH=$1
    UPLOAD_FILE=$2
    UPLOAD_FILE="s3://craftassist/pubr/checksums/${UPLOAD_FILE}"
    echo "Uploading $CHECKSUM_PATH $(cat $CHECKSUM_PATH) to $UPLOAD_FILE"
    
    aws s3 cp $CHECKSUM_PATH $UPLOAD_FILE
}

process_chksm() {
    CHECKSUM_PATH=$1
    CHECKSUM_FOLDER=$2
    UPLOAD_FILE=$3
    find $CHECKSUM_FOLDER -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $CHECKSUM_PATH
    upload_path_to_s3 $CHECKSUM_PATH $UPLOAD_FILE
}

echo "Computing hashes ..."
if [ "$HASH_PATH" = "models" ]
then
    if [ $AGENT == "locobot" ]; then
        process_chksm "$HASH_PATH/locobot_checksum.txt" "models/perception" "locobot_checksum.txt"
    else # craftassist
        process_chksm "$HASH_PATH/checksum.txt" "models/semantic_parser" "nsp_checksum.txt"
    fi
else # datasets
    process_chksm "$HASH_PATH/checksum.txt" $HASH_PATH/ "datasets_checksum.txt"
fi

