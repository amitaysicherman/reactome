#!/bin/sh
#SBATCH --time=0-3
#SBATCH --array=1-45
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p protein_drug/train_eval_all.txt)
python3 protein_drug/train_eval.py $args