#!/bin/bash

# slurm
PARTITION="priority"
NGPUS=8
CONSTRAINT=""
NODES=1

# root dir for datasets
DATA_ROOT="/checkpoint/yuxuans/datasets/inst_seg/" 
DATA_DIR="D11" 
DATA_PATH="$DATA_ROOT$DATA_DIR/"

# hyperparameters
NUM_WORKERS=8
NEPOCHS=4000
NOTE="HITL_VISION_EXP_8"
BATCH_SZ=1600
VALID_BATCH_SZ=1000

# LRS=(0.001 0.01)
# SAMPLE_EMPTY_PROBS=(0.005 0.05)
# HIDDEN_DIMS=(128)
# NO_TARGET_PROBS=(0.3 0.5)
# PROB_THRESHOLDS=(0.3 0.5 0.8)
# QUERY_EMBEDS=("clip")

# testing
LRS=(0.001 0.01)
SAMPLE_EMPTY_PROBS=(0.05 0.005)
HIDDEN_DIMS=(128)
NO_TARGET_PROBS=(0.3)
PROB_THRESHOLDS=(0.8 0.5)
QUERY_EMBEDS=("clip")

for LR in "${LRS[@]}"
do 
    for SAMPLE_EMPTY_PROB in "${SAMPLE_EMPTY_PROBS[@]}"
    do
        for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"
        do
            for NO_TARGET_PROB in "${NO_TARGET_PROBS[@]}"
            do
                for PROB_THRESHOLD in "${PROB_THRESHOLDS[@]}"
                do
                    for QUERY_EMBED in "${QUERY_EMBEDS[@]}"
                    do
                        JOB_NAME="data_${DATA_DIR}_nepochs_${NEPOCHS}_numworkers_${NUM_WORKERS}_lr_${LR}_batchsz_${BATCH_SZ}_validbatchsize_${VALID_BATCH_SZ}_sampleEmptyProb_${SAMPLE_EMPTY_PROB}_hiddenDim_${HIDDEN_DIMS}_noTargetProb_${NO_TARGET_PROB}_probThreshold_${PROB_THRESHOLD}_queryEmbed_${QUERY_EMBED}_note_${NOTE}"
                        ./slurm/launch.sh $JOB_NAME $PARTITION $NGPUS "" $NODES \
                        --num_epochs=$NEPOCHS \
                        --batchsize=$BATCH_SZ \
                        --valid_batchsize=$VALID_BATCH_SZ \
                        --lr=$LR \
                        --cuda \
                        --distributed \
                        --sample_empty_prob=$SAMPLE_EMPTY_PROB \
                        --hidden_dim=$HIDDEN_DIM \
                        --no_target_prob=$NO_TARGET_PROB \
                        --run_name=$JOB_NAME \
                        --prob_threshold=$PROB_THRESHOLD \
                        --data_dir=$DATA_PATH \
                        --query_embed=$QUERY_EMBED \
                        --num_workers=$NUM_WORKERS
                    done
                done
            done
        done
    done
done