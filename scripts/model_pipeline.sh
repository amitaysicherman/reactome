#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
args=(
    "--fuse_name default_args --gnn_fuse_name default_args --gnn_name default_args --eval_model_name default_args"
)

for i in "${!args[@]}"; do
    gpu_index=$((i % num_gpus))
    arg_i="${args[$i]}"
    script="python model/contrastive_learning.py $arg_i && python model/train_gnn.py $arg_i && python model/eval_model.py $arg_i"
    echo $script

    CUDA_VISIBLE_DEVICES="${gpus[$gpu_index]}" $script &
    if (( (i + 1) % num_gpus == 0 )); then
        wait
    fi
done
wait
