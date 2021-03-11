#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


function pyabspath() {
    python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" $1
}

ROOTDIR=$(pyabspath $(dirname "$0")/../../)
echo "$ROOTDIR"
MODELS_DIRNAME=models_folder

if [ -z $1 ]
then
	AGENT="craftassist"
else
	AGENT=$1
fi
CHKSUM=$2

echo "Checksum" $CHKSUM

MODELS_DIRNAME=models_folder

cd $ROOTDIR

echo "====== Downloading http://craftassist.s3-us-west-2.amazonaws.com/pubr/${MODELS_DIRNAME}_${CHKSUM}.tar.gz to $ROOTDIR/${MODELS_DIRNAME}_${CHKSUM}.tar.gz ======"
curl http://craftassist.s3-us-west-2.amazonaws.com/pubr/${MODELS_DIRNAME}_${CHKSUM}.tar.gz -o $MODELS_DIRNAME.tar.gz

if [ -d "${AGENT}/agent/models" ]
then
	echo "Overwriting models directory"
	rm -rf $AGENT/agent/models/
fi

mkdir -p $AGENT/agent/models/

tar -xzvf $MODELS_DIRNAME.tar.gz -C $AGENT/agent/models/ --strip-components 1

if [ $AGENT == "locobot" ]; then
    curl https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_models.tar.gz -o locobot_models.tar.gz
    tar -xzvf locobot_models.tar.gz -C $AGENT/agent/models/
	curl https://locobot-bucket.s3-us-west-2.amazonaws.com/perception_test_assets.tar.gz | tar -xzv -C $AGENT/test/test_assets/
fi