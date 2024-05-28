#!/bin/bash

#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
scripts=(
    "python train_gnn.py --pretrained_method 0 --fuse_name mp_recon --train_all_emd 0"
    "python train_gnn.py --pretrained_method 1 --fuse_name mp_recon --train_all_emd 0"
    "python train_gnn.py --pretrained_method 2 --fuse_name all_recon --train_all_emd 0"
    "python train_gnn.py --pretrained_method 2 --fuse_name mp_recon --train_all_emd 0"
    "python train_gnn.py --pretrained_method 2 --fuse_name mp --train_all_emd 0"
    "python train_gnn.py --pretrained_method 2 --fuse_name all --train_all_emd 0"
    "python train_gnn.py --pretrained_method 0 --fuse_name mp_recon --train_all_emd 1"
    "python train_gnn.py --pretrained_method 1 --fuse_name mp_recon --train_all_emd 1"
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

