#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-63
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p torchdrug_tasks/prep_all.txt)
python torchdrug_tasks/prep.py $args
