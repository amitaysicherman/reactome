#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-15
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/run_full_conf.txt)
python3 scripts/run_full_conf.py $args --db_dataset Davis
python3 scripts/run_full_conf.py $args --db_dataset KIBA

