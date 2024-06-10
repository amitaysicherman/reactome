#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
args=(
  "--name trans_fuse_recon --gnn_conv_type TransformerConv --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_recon 1"
  "--name trans_fuse_all_to_one --gnn_conv_type TransformerConv --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one all"
  "--name trans_no_pretrained_trained --gnn_conv_type TransformerConv --gnn_pretrained_method 0 --gnn_train_all_emd 1"
  "--name trans_pretrained_freeze --gnn_conv_type TransformerConv --gnn_pretrained_method 1 --gnn_train_all_emd 0"
  "--name large_fuse_recon --gnn_hidden_channels 1024 --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_recon 1"
  "--name large_fuse_all_to_one --gnn_hidden_channels 1024 --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one all"
  "--name large_no_pretrained_trained --gnn_hidden_channels 1024 --gnn_pretrained_method 0 --gnn_train_all_emd 1"
  "--name large_pretrained_freeze --gnn_hidden_channels 1024 --gnn_pretrained_method 1 --gnn_train_all_emd 0"
  "--name small_fuse_recon --gnn_hidden_channels 32 --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_recon 1"
  "--name small_fuse_all_to_one --gnn_hidden_channels 32 --gnn_pretrained_method 2 --gnn_train_all_emd 0 --fuse_all_to_one all"
  "--name small_no_pretrained_trained --gnn_hidden_channels 32 --gnn_pretrained_method 0 --gnn_train_all_emd 1"
  "--name small_pretrained_freeze --gnn_hidden_channels 32 --gnn_pretrained_method 1 --gnn_train_all_emd 0"

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
