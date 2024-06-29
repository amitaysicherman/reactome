#!/bin/sh
#SBATCH --time=0-3
#SBATCH --array=1-162
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_seq_const.txt)
python3 dataset/seq_const.py $args