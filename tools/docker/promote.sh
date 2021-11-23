#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


set -e


if [ $# == 2 ]; then
    IMAGE_SHA=$1
    IMAGE_TAG=$2

else
    echo "Usage: $0 <img sha1> <tag>"
    exit 1
fi

MANIFEST=$(aws ecr batch-get-image \
    --repository-name craftassist \
    --region us-west-1 \
    --image-ids imageDigest=$IMAGE_SHA \
    --query "images[].imageManifest" \
    --output text)

echo $MANIFEST

aws ecr put-image \
    --repository-name craftassist \
    --image-tag $IMAGE_TAG \
    --image-manifest "$MANIFEST" \
    --region us-west-1    

echo
echo Success!
echo "Tagged Image <$IMAGE_SHA> as <$IMAGE_TAG>"
