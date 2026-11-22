#!/bin/bash

CUDA_ID=$1

for seed in {16..16}
do
    python3 main.py \
        --dataset=MSL \
        --data_path=./data/msl \
        --seed=${seed} \
        --device=cuda:$CUDA_ID \
        --lr=5e-4 > /dev/null 2>&1 &
done