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
    step_size=$(echo "scale=6; $base * $decay^$i" | bc -l)
    step_sizes+=($step_size)
done

echo "Step sizes: ${step_sizes[@]}"
exp_times=("seg_512" "seg_256" "seg_128" "seg_64")

for (( i=1; i<=num; i++ )); do
    exp_time=${exp_times[$i]}
    step_size=${step_sizes[$i]} 

    mkdir -p logs_txt/${exp_time}/blender/seg

    for val in "${StringArray[@]}"; do
        echo "Executing command for $val with step_size=${step_size} and exp_time=${exp_time}"
        $PYTHON examples/train_segngp_nerf_occ.py \
        --train_steps 50000 \
        --scene "$val" \
        --exp_time $exp_time \
        --data_root data/nerf_synthetic \
        --version v1 \
        --render_step_size ${step_size} >> "logs_txt/${exp_time}/blender/seg/${val}_output_seg.txt"
    done

    echo "All commands for step_size=${step_size} and exp_time=${exp_time} executed. Check outputs folder for output files."
done
