#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-6
#SBATCH --requeue
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH --mem=64G

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_fuse.txt)
python3 model/contrastive_learning.py $args