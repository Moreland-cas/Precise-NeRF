#!/bin/bash

PYTHON=/data/zhangboyuan/ProgramFiles/anaconda3/envs/nerfacc/bin/python
export CUDA_VISIBLE_DEVICES=0

declare -a StringArray=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

base="0.005"  
num=1
decay="2."

declare -a step_sizes
step_sizes=()
for ((i=1; i<=num; i++)); do
    # calculate base * (decay)^i
    step_size=$(echo "scale=6; $base * $decay^$i" | bc -l)
    step_sizes+=($step_size)
done

echo "Step sizes: ${step_sizes[@]}"
exp_times=("vanilla_512" "vanilla_256" "vanilla_128" "vanilla_64")

for (( i=1; i<=num; i++ )); do
    exp_time=${exp_times[$i]}
    step_size=${step_sizes[$i]} 

    mkdir -p logs_txt/${exp_time}/blender/vanilla

    for val in "${StringArray[@]}"; do
        echo "Executing command for $val with step_size=${step_size} and exp_time=${exp_time}"
        $PYTHON examples/train_ngp_nerf_occ.py \
        --train_steps 50000 \
        --scene "$val" \
        --exp_time $exp_time \
        --data_root data/nerf_synthetic \
        --version v1 \
        --render_step_size ${step_size} >> "logs_txt/${exp_time}/blender/vanilla/${val}_output.txt"
    done

    echo "All commands for step_size=${step_size} and exp_time=${exp_time} executed. Check outputs folder for output files."
done
