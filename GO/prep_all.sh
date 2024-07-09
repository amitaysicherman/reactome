#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-15
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p GO/prep_all.txt)
python3 GO/prep.py $args --self_token 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