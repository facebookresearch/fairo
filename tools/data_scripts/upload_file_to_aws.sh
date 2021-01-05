#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

ROOTDIR=$(readlink -f $(dirname "$0")/../../)
echo "$ROOTDIR"

CRAFTASSIST_PATH="${ROOTDIR}/craftassist/agent/"
DATA_SCRIPTS_PATH="tools/data_scripts/compute_and_upload_checksum.sh"
echo "$CRAFTASSIST_PATH"
echo "$DATA_SCRIPTS_PATH"

declare -a DIRLIST
if [ -z $1 ]
then
    # Default to uploading both models and datasets
	DIRLIST=(models datasets)
else
	DIRLIST=($1)
fi

echo "Uploading to S3: ${DIRLIST[*]}"

for dirname in "${DIRLIST[@]}" 
do 
	DEST_PATH="${CRAFTASSIST_PATH}${dirname}/"
	echo "$DEST_PATH"
	echo "====== Compressing models directory ${DEST_PATH} ======"

	cd $CRAFTASSIST_PATH

	tar --exclude='*/\.*' --exclude='*checksum*' ${dirname}/ -czvf ${dirname}_folder.tar.gz

	echo "tar file created at ${PWD}/${dirname}_folder.tar.gz"

	echo "====== Uploading models and datasets to S3 ======"
	aws s3 cp ${CRAFTASSIST_PATH}${dirname}_folder.tar.gz s3://craftassist/pubr/
        cd ../..
	./$DATA_SCRIPTS_PATH craftassist $dirname
done
