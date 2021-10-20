#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script creates a tar and hash of the models directory.
# If uploading files to S3 through console UI, go to the web interface at: 
# https://s3.console.aws.amazon.com/s3/buckets/craftassist?region=us-west-2&prefix=pubr/&showversions=false 
# and upload ``models_folder_<sha1sum>.tar.gz``.

function pyabspath() {
    python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" $1
}

ROOTDIR=$(pyabspath $(dirname "$0")/../../)
echo "$ROOTDIR"
. ${ROOTDIR}/tools/data_scripts/checksum_fn.sh --source-only

if [ -z $1 ]
then
    AGENT="craftassist"
    echo "Agent name not specified, defaulting to craftassist"
else
	AGENT=$1
fi

AGENT_PATH="${ROOTDIR}/agents/${AGENT}/"
echo "$AGENT_PATH"

cd $AGENT_PATH
CHECKSUM_PATH="models/checksum.txt"
calculate_sha1sum "${AGENT_PATH}models/semantic_parser" "${AGENT_PATH}${CHECKSUM_PATH}"

CHKSUM=$(cat $CHECKSUM_PATH)
echo "CHECKSUM" $CHKSUM

tar -czvf models_folder_${CHKSUM}.tar.gz --exclude='*/\.*' --exclude='*checksum*' models/semantic_parser

read -p "Do you want to upload models_folder_${CHKSUM}.tar.gz to S3 ? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Uploading ..."
    aws s3 cp models_folder_${CHKSUM}.tar.gz s3://craftassist/pubr/
fi

cd $AGENT_PATH
CHECKSUM_PATH="models/perception_checksum.txt"
calculate_sha1sum "${AGENT_PATH}models/perception" "${AGENT_PATH}${CHECKSUM_PATH}"

CHKSUM=$(cat $CHECKSUM_PATH)
echo "CHECKSUM" $CHKSUM

tar -czvf craftassist_perception_${CHKSUM}.tar.gz --exclude='*/\.*' --exclude='*checksum*' models/perception

read -p "Do you want to upload craftassist_perception_${CHKSUM}.tar.gz to S3 ? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Uploading ..."
    aws s3 cp craftassist_perception_${CHKSUM}.tar.gz s3://craftassist/pubr/
fi