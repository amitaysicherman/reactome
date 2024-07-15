#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=186-200
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/run_full_conf.txt)
python3 scripts/run_full_conf.py $args

