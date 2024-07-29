#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_td.txt)
python torchdrug_tasks/hyperparameter_search.py $args
