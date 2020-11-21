#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script computes hashes for local directories and uploads them to AWS. Used to store hashes for the deployed models.
ROOTDIR=$(readlink -f $(dirname "$0")/../../)
echo "$ROOTDIR"

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

CHECKSUM_PATH="$HASH_PATH/checksum.txt"

echo "Computing hashes for ${AGENT}/agent/${HASH_PATH}/"
if [ "$HASH_PATH" = "models" ]
then
    if [ $AGENT == "locobot" ]; then
        find models/semantic_parser/ models/perception -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $CHECKSUM_PATH
    else
        find models/semantic_parser/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $CHECKSUM_PATH
    fi
else # datasets
    find $HASH_PATH/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $CHECKSUM_PATH
fi

echo $CHECKSUM_PATH
cat $CHECKSUM_PATH

UPLOAD_FILE="s3://craftassist/pubr/${AGENT}_${HASH_PATH}_checksum.txt"
echo "Uploading to $UPLOAD_FILE"
aws s3 cp $CHECKSUM_PATH $UPLOAD_FILE --acl public-read
