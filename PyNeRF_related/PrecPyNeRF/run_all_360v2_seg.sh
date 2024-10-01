#!/bin/bash

PYTHON=/data/zhangboyuan/ProgramFiles/anaconda3/envs/precpynerf/bin/python
export CUDA_VISIBLE_DEVICES=0

declare -a StringArray=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump" "flowers" "treehill")

for val in "${StringArray[@]}"; do
    echo "Executing command for $val"
    ns-train pynerf \
    --output-dir ./outputs_360 \
    --experiment-name "$val" \
    --data ./data/360_v2/"$val" \
    --viewer.websocket-port-default 6006 mipnerf360-data &
    PID=$!
    sleep 200m
    kill -INT $PID
done

echo "All commands executed. Check outputs folder for output files."
