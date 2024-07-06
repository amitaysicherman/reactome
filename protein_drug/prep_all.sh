#!/bin/sh
#SBATCH --time=0-6
#SBATCH --array=1-24
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p protein_drug/prep_all.txt)
python3 protein_drug/prep.py $args --skip_if_exists 1 --self_token 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