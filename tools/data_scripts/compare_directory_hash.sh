#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script checks if models and datasets are up to date, and either triggers a download or gives the user a warning to update local files.
function pyabspath() {
    python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" $1
}

ROOTDIR=$(pyabspath $(dirname "$0")/../../)
echo "$ROOTDIR"

if [ -z $1 ]
then
    AGENT="craftassist"
    echo "Agent name not specified, defaulting to craftassist"
else
    AGENT=$1
fi

AGENT_PATH="${ROOTDIR}/${AGENT}/agent/"
echo "$AGENT_PATH"

# in case directories don't even exist, create them
mkdir -p $AGENT_PATH/datasets
mkdir -p $AGENT_PATH/models
mkdir -p $AGENT_PATH/models/semantic_parser
mkdir -p $AGENT_PATH/models/perception


compare_checksum() {
    LOCAL_CHKSM=$1
    FOLDER=$2
    echo "Comparing $FOLDER checksums - Local " $(cat $LOCAL_CHKSM) "AWS " $(cat ${FOLDER}_checksum.txt)
    curl "http://craftassist.s3-us-west-2.amazonaws.com/pubr/checksums/${FOLDER}_checksum.txt" -o ${FOLDER}_checksum.txt
    if cmp -s $LOCAL_CHKSM ${FOLDER}_checksum.txt
    then
        echo "Local $FOLDER directory is up to date."
    else
	    try_download $FOLDER
    fi
}

try_download() {
    FOLDER=$1
    echo "*********************************************************************************************"
    echo "Local ${FOLDER} directory is out of sync. Downloading latest. Use --dev to disable downloads."
    echo "*********************************************************************************************"
    echo "Downloading ${FOLDER} directory"
    SCRIPT_PATH="$ROOTDIR/tools/data_scripts/fetch_aws_models.sh"
    if cmp -s $FOLDER "datasets"
    then
        SCRIPT_PATH="$ROOTDIR/tools/data_scripts/fetch_aws_datasets.sh"
    fi
    echo "Downloading using script " $SCRIPT_PATH
    "$SCRIPT_PATH" "$AGENT"
}

pushd $AGENT_PATH

# Comparing hashes for local directories
# Default models and datasets shared by all agents
find models/semantic_parser -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > models/nsp_checksum.txt
cat "models/nsp_checksum.txt"
compare_checksum "models/nsp_checksum.txt" "nsp" 

find datasets -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > datasets/checksum.txt
compare_checksum "datasets/checksum.txt" "datasets"

# Agent specific models 
if [ $AGENT == "locobot" ]; then
    find models/perception -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > models/locobot_checksum.txt
    compare_checksum "models/locobot_checksum.txt" "locobot"
fi
