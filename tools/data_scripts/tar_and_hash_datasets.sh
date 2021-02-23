#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This script creates a tar and hash of the datasets directory used in craftassist.
# This is an alternative to ``compute_and_upload_checksum.sh`` that does not require awscli setup.
# The resulting tar and checksum would be in
# python/craftassist/datasets_folder.tar.gz
# python/craftassist/datasets/checksum.txt
# If uploading files to S3 through console UI, go to the web interface at: 
# https://s3.console.aws.amazon.com/s3/buckets/craftassist?region=us-west-2&prefix=pubr/&showversions=false 
# and upload both ``datasets_folder.tar.gz`` and ``checksum.txt``.

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

CRAFTASSIST_PATH="${ROOTDIR}/${AGENT}/agent/"
echo "$CRAFTASSIST_PATH"

DIRNAME="datasets"

cd $CRAFTASSIST_PATH
tar -czvf ${DIRNAME}_folder.tar.gz --exclude='*/\.*' --exclude='*checksum*' ${DIRNAME}/

CHECKSUM_PATH="${DIRNAME}/checksum.txt"
find ${DIRNAME} -type f ! -name '*checksum*' -not -path '*/\.*' -print0 | sort -z | xargs -0 sha1sum | sha1sum > $CHECKSUM_PATH

cat $CHECKSUM_PATH

aws s3 cp ${DIRNAME}_folder.tar.gz s3://craftassist/pubr/

UPLOAD_FILE="s3://craftassist/pubr/checksums/datasets_checksum.txt"

aws s3 cp $CHECKSUM_PATH $UPLOAD_FILE

