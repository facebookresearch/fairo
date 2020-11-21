#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script checks if models and datasets are up to date, and either triggers a download or gives the user a warning to update local files.
ROOTDIR=$(readlink -f $(dirname "$0")/../../)
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

# Compute hashes for local directories
echo "Computing hashes for ${AGENT_PATH}/models/ and ${AGENT_PATH}/datasets/"
pushd $AGENT_PATH

if [ $AGENT == "locobot" ]; then
    find models/semantic_parser/ models/perception -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > models/checksum.txt
else # minecraft
    find models/semantic_parser/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > models/checksum.txt
fi
find datasets/ -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > datasets/checksum.txt
pushd ../..


echo
# Download AWS checksum
echo "Downloading latest hash from AWS"

curl "http://craftassist.s3-us-west-2.amazonaws.com/pubr/${AGENT}_models_checksum.txt" -o $AGENT/agent/models/models_checksum_aws.txt
echo "Latest model hash in $AGENT/models/models_checksum_aws.txt"
curl "http://craftassist.s3-us-west-2.amazonaws.com/pubr/${AGENT}_datasets_checksum.txt" -o $AGENT/agent/datasets/datasets_checksum_aws.txt
echo "Latest data hash in $AGENT/datasets/datasets_checksum_aws.txt"
echo

if cmp -s $AGENT/agent/models/checksum.txt $AGENT/agent/models/models_checksum_aws.txt
then
	echo "Local models directory is up to date."
else
	echo "Local models directory is out of sync. Would you like to download the updated files from AWS? This overwrites ${AGENT}/agent/models/"
	read -p "Enter Y/N: " permission
	echo $permission
	if [ "$permission" == "Y" ] || [ "$permission" == "y" ] || [ "$permission" == "yes" ]; then
		echo "Downloading models directory"
		SCRIPT_PATH="$ROOTDIR/tools/data_scripts/fetch_aws_models.sh"
		echo $SCRIPT_PATH
		"$SCRIPT_PATH" "$AGENT"
	else
		echo "Warning: Outdated models can cause breakages in the repo."
	fi
fi

if cmp -s $AGENT/agent/datasets/checksum.txt $AGENT/agent/datasets/datasets_checksum_aws.txt
then
        echo "Local datasets directory is up to date."
else
        echo "Local datasets directory is out of sync. Would you like to download the updated files from AWS? This overwrites ${AGENT}/agent/datasets/"
        read -p "Enter Y/N: " permission
        echo $permission
        if [ "$permission" == "Y" ] || [ "$permission" == "y" ] || [ "$permission" == "yes" ]; then
                echo "Downloading datasets directory"
                SCRIPT_PATH="$ROOTDIR/tools/data_scripts/fetch_aws_datasets.sh"
                echo $SCRIPT_PATH
                "$SCRIPT_PATH" "$AGENT"
                exit 1
        else
                echo "Warning: Outdated datasets can cause breakages in the repo."
                exit 1
        fi
fi
