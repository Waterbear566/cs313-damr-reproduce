#!/bin/bash

CUDA_ID=$1 

for seed in {16..16}
do
    python3 main.py \
        --dataset=SMD \
        --data_path=./data/smd \
        --seed=${seed} \
        --device=cuda:$CUDA_ID \
        --lr=1e-6 > /dev/null 2>&1 &
done