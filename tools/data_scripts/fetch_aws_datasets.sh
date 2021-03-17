#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

function pyabspath() {
    python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" $1
}

ROOTDIR=$(pyabspath $(dirname "$0")/../../)
echo "$ROOTDIR"

if [ -z $1 ]
then
	AGENT="craftassist"
	echo "Defaulting to default agent: '$AGENT'"
else
	AGENT=$1
fi

if [ -z $2 ]
then
	CHECKSUM_FILE="tools/data_scripts/default_checksums/datasets.txt"
	CHECKSUM=`cat $CHECKSUM_FILE`  
	echo "Downloading datasets folder with default checksum from file: '$CHECKSUM_FILE'"
else
	CHECKSUM=$2
fi

echo "Checksum" $CHECKSUM

DATA_DIRNAME=datasets_folder

cd $ROOTDIR

echo "====== Downloading  http://craftassist.s3-us-west-2.amazonaws.com/pubr/${DATA_DIRNAME}_${CHECKSUM}.tar.gz to $ROOTDIR/$DATA_DIRNAME.tar.gz ======"
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/${DATA_DIRNAME}_${CHECKSUM}.tar.gz -o $DATA_DIRNAME.tar.gz

if [ -d "${AGENT}/agent/datasets" ]
then
	echo "Overwriting datasets directory"
	rm -rf $AGENT/agent/datasets/
fi
mkdir -p $AGENT/agent/datasets/

tar -xzvf $DATA_DIRNAME.tar.gz -C $AGENT/agent/datasets/ --strip-components 1
