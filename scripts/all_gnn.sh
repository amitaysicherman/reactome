#!/bin/sh
#SBATCH --time=0-3
#SBATCH --array=1-8
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_gnn.txt)
python3 model/train_gnn.py $args