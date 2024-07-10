#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-3
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p mol_tasks/prep_all.txt)
python3 mol_tasks/prep.py $args --self_token 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