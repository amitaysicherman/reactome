#!/bin/bash

#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
scripts=(
    "python model_tags.py --pretrained_method 0 --fuse_config 8192_0.0_512_0.001_1_1024_1_1 --train_all_emd 0"
    "python model_tags.py --pretrained_method 1 --fuse_config 8192_0.0_512_0.001_1_1024_1_1 --train_all_emd 0"
    "python model_tags.py --pretrained_method 2 --fuse_config 8192_0.0_512_0.001_1_1024_0_0 --train_all_emd 0"
    "python model_tags.py --pretrained_method 2 --fuse_config 8192_0.0_512_0.001_1_1024_1_1 --train_all_emd 0"
    "python model_tags.py --pretrained_method 2 --fuse_config 8192_0.0_512_0.001_1_1024_1_0 --train_all_emd 0"
    "python model_tags.py --pretrained_method 2 --fuse_config 8192_0.0_512_0.001_1_1024_0_1 --train_all_emd 0"
    "python model_tags.py --pretrained_method 0 --fuse_config 8192_0.0_512_0.001_1_1024_1_1 --train_all_emd 1"
    "python model_tags.py --pretrained_method 1 --fuse_config 8192_0.0_512_0.001_1_1024_1_1 --train_all_emd 1"
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

