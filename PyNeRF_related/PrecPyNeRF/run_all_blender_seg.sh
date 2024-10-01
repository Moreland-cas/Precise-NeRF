#!/bin/bash

PYTHON=/data/zhangboyuan/ProgramFiles/anaconda3/envs/precpynerf/bin/python
export CUDA_VISIBLE_DEVICES=0

declare -a StringArray=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for val in "${StringArray[@]}"; do
    echo "Executing command for $val - Iteration $i"
    ns-train pynerf-occupancy-grid \
    --output-dir ./outputs_seg \
    --experiment-name "$val" \
    --data ./data/multiscale/"$val" \
    --viewer.websocket-port-default 6006 &
    PID=$!
    sleep 50m
    kill -INT $PID
done

echo "All commands executed. Check outputs folder for output files."


