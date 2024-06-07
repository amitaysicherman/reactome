#!/bin/bash

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
scripts=(
    "python model/train_gnn.py --gnn_pretrained_method 0 --gnn_fuse_name mp_recon --gnn_train_all_emd 0 --gnn_name no_pretrain_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 0 --gnn_fuse_name mp_recon --gnn_train_all_emd 0 --gnn_name no_pretrain_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 1 --gnn_fuse_name mp_recon --gnn_train_all_emd 0 --gnn_name pretrain_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 2 --gnn_fuse_name all_recon --gnn_train_all_emd 0 --gnn_name fuse_all_recon_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 2 --gnn_fuse_name mp_recon --gnn_train_all_emd 0 --gnn_name fuse_mp_recon_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 2 --gnn_fuse_name mp --gnn_train_all_emd 0 --gnn_name fuse_mp_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 2 --gnn_fuse_name all --gnn_train_all_emd 0 --gnn_name fuse_all_freeze"
    "python model/train_gnn.py --gnn_pretrained_method 0 --gnn_fuse_name mp_recon --gnn_train_all_emd 1 --gnn_name no_pretrain_train"
    "python model/train_gnn.py --gnn_pretrained_method 1 --gnn_fuse_name mp_recon --gnn_train_all_emd 1 --gnn_name pretrain_train"
)

for i in ${!scripts[@]}; do
    gpu_index=$((i % num_gpus))
    CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} ${scripts[$i]} &
    # If all GPUs are in use, wait for any to finish before continuing
    if (( (i + 1) % num_gpus == 0 )); then
        wait
    fi
done
wait

