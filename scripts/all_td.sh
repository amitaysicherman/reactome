#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-20
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --nodes=20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_td.txt)
python torchdrug_tasks/hyperparameter_search.py $args
