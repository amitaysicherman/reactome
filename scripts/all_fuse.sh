#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-6
#SBATCH --requeue
#SBATCH --mem=8G

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_fuse.txt)
python python model/contrastive_learning.py "${args}"