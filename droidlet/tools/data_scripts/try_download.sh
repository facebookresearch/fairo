#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script checks if models and datasets are up to date, and downloads default 
# assets (specified in `tool/data_scripts/default_checksums`) if they are stale.

function pyabspath() {
    python3 -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" $1
}

ROOTDIR=$(pyabspath $(dirname "$0")/../../../)
echo "Rootdir ${ROOTDIR}"

# Optionally fetch secure resources for internal users and prod systems
python3 ${ROOTDIR}/droidlet/tools/data_scripts/fetch_internal_resources.py ${ROOTDIR}/droidlet/documents/internal/safety.txt

. ${ROOTDIR}/droidlet/tools/data_scripts/checksum_fn.sh --source-only # import checksum function

if [ -z $1 ]
then
    AGENT="craftassist"
    echo "Agent name not specified, defaulting to craftassist"
else
    AGENT=$1
fi

AGENT_PATH="${ROOTDIR}/agents/${AGENT}/"
echo "agent path ${AGENT_PATH}"

# in case directories don't even exist, create them
mkdir -p $AGENT_PATH/datasets
mkdir -p $AGENT_PATH/models
mkdir -p $AGENT_PATH/models/nlu
mkdir -p $AGENT_PATH/models/perception

compare_checksum_try_download() {
    LOCAL_CHKSM=$(cat $1)
    ASSET=$2
    LATEST_CHKSM=$(cat ${ROOTDIR}/droidlet/tools/data_scripts/default_checksums/${ASSET}.txt)
    echo "Comparing $ASSET checksums" 
    echo "Local " $LOCAL_CHKSM
    echo "Latest " $LATEST_CHKSM
    if [[ "$LOCAL_CHKSM" == "$LATEST_CHKSM" ]]; then
        echo "Local $ASSET asset is up to date."
    else
	    try_download $ASSET $LATEST_CHKSM
    fi
}

try_download() {
    ASSET=$1
    CHKSUM=$2
    echo "*********************************************************************************************"
    echo "Local ${ASSET} asset is out of sync. Downloading latest. Use --dev to disable downloads."
    echo "*********************************************************************************************"
    echo "Downloading ${ASSET} asset"
    SCRIPT_PATH="$ROOTDIR/droidlet/tools/data_scripts/fetch_aws_models.sh"
    if cmp -s $ASSET "datasets"
    then
        SCRIPT_PATH="$ROOTDIR/droidlet/tools/data_scripts/fetch_aws_datasets.sh"
    fi
    echo "Downloading using script " $SCRIPT_PATH
    "$SCRIPT_PATH" "$AGENT" "$CHKSUM" "$ASSET"
}

pushd $AGENT_PATH

# Comparing hashes for local directories
# Default models and datasets shared by all agents
# Remove existing checksum files so that they can be re-calculated
rm ${AGENT_PATH}models/*checksum.txt

calculate_sha1sum "${AGENT_PATH}models/nlu" "${AGENT_PATH}models/nsp_checksum.txt"
compare_checksum_try_download "models/nsp_checksum.txt" "nsp" 

calculate_sha1sum "${AGENT_PATH}datasets" "${AGENT_PATH}datasets/checksum.txt"
compare_checksum_try_download "${AGENT_PATH}datasets/checksum.txt" "datasets"

# Agent specific models 
if [ $AGENT == "locobot" ]; then
    calculate_sha1sum "${AGENT_PATH}models/perception" "${AGENT_PATH}models/locobot_perception_checksum_local.txt"
    compare_checksum_try_download "${AGENT_PATH}models/locobot_perception_checksum_local.txt" "locobot_perception"
fi

# Craftassist specific perception models 
if [ $AGENT == "craftassist" ]; then
    calculate_sha1sum "${AGENT_PATH}models/perception" "${AGENT_PATH}models/craftassist_perception_checksum_local.txt"
    compare_checksum_try_download "${AGENT_PATH}models/craftassist_perception_checksum_local.txt" "craftassist_perception"
fi
