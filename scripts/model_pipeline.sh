#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
args=(
  "--name no_pretrained_freeze --gnn_pretrained_method 0 --gnn_train_all_emd 0"
  "--name no_pretrained_trained --gnn_pretrained_method 0 --gnn_train_all_emd 1"
  "--name pretrained_freeze --gnn_pretrained_method 1 --gnn_train_all_emd 0"
  "--name pretrained_trained --gnn_pretrained_method 1 --gnn_train_all_emd 1"
  "--name fuse --gnn_pretrained_method 2 --gnn_train_all_emd 0"
  "--name fuse_recon --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_recon 1"
  "--name fuse_all_to_one --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one all"
  "--name fuse_all_to_port --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one protein"
  "--name fuse_all_to_mol --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one molecule"
  "--name fuse_all_to_text --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one text"
  "--name fuse_all_to_dna --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one dna"
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
