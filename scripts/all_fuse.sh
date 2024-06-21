#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-6
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_fuse.txt)
python3 model/contrastive_learning.py $args