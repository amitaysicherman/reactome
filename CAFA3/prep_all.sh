#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-5
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p CAFA3/prep_all.txt)
python3 CAFA3/prep.py $args --self_token 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