#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
scripts=(
    "python model/contrastive_learning.py --proteins_molecules_only 0 --recon 0 --name mp"
    "python model/contrastive_learning.py --proteins_molecules_only 0 --recon 1 --name mp_recon"
    "python model/contrastive_learning.py --proteins_molecules_only 1 --recon 0 --name all"
    "python model/contrastive_learning.py --proteins_molecules_only 1 --recon 1 --name all_recon"
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



