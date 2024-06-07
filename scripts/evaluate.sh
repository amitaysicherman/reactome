#!/bin/bash

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .



gpus=($(seq 0 $((num_gpus - 1))))
scripts=(
    "python model/eval_model.py --eval_model_name gnn_no_pretrain_freeze"
    "python model/eval_model.py --eval_model_name gnn_pretrain_freeze"
    "python model/eval_model.py --eval_model_name gnn_fuse_all_recon_freeze"
    "python model/eval_model.py --eval_model_name gnn_fuse_mp_recon_freeze"
    "python model/eval_model.py --eval_model_name gnn_fuse_mp_freeze"
    "python model/eval_model.py --eval_model_name gnn_fuse_all_freeze"
    "python model/eval_model.py --eval_model_name gnn_no_pretrain_train"
    "python model/eval_model.py --eval_model_name gnn_pretrain_train"
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

