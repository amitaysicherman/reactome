#!/bin/sh
#SBATCH --time=0-3
#SBATCH --array=1-24
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_fuse.txt)
python3 model/contrastive_learning.py $args