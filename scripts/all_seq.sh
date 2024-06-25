#!/bin/sh
#SBATCH --time=0-3
#SBATCH --array=1-54
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_seq.txt)
python3 dataset/seq_dataset.py $args
