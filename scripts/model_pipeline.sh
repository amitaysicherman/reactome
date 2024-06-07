#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
args=(
    "--fuse_name tmp --gnn_fuse_name tmp --gnn_name tmp --eval_model_name tmp --fuse_epochs 2 --gnn_epochs 2"
)

for i in "${!args[@]}"; do
    gpu_index=$((i % num_gpus))
    arg_i="${args[$i]}"
    script="python model/contrastive_learning.py $arg_i && python model/train_gnn.py $arg_i && python model/eval_model.py $arg_i"
    echo $script

    CUDA_VISIBLE_DEVICES="${gpus[$gpu_index]}" bash -c "$script" &
    if (( (i + 1) % num_gpus == 0 )); then
        wait
    fi
done
wait
