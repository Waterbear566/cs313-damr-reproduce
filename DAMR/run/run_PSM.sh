#!/bin/bash

CUDA_ID=$1

for seed in {16..16}
do
    python3 main.py \
        --dataset=PSM \
        --data_path=./data/psm \
        --seed=${seed} \
        --device=cuda:$CUDA_ID \
        --lr=1e-4 > /dev/null 2>&1 &
done