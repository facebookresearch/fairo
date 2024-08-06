#!/bin/bash

if [ ! -f "./third_party/graspnet-baseline/data/graspnet/checkpoint-rs.tar" ]; then
    gdown --id 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O ./third_party/graspnet-baseline/data/graspnet/
else
    echo "Already have checkpoints for graspnet-baseline"
fi

if [ ! -d "./third_party/UnseenObjectClustering/data/checkpoints" ]; then
    gdown --id 1O-ymMGD_qDEtYxRU19zSv17Lgg6fSinQ -O ./third_party/UnseenObjectClustering/data/
    cd third_party/UnseenObjectClustering/data/
    unzip ./checkpoints.zip
else
    echo "Already have checkpoints for UnseenObjectClustering"
fi
