#!/bin/sh
#SBATCH --time=1-0
#SBATCH --array=1-6
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 24
#SBATCH --requeue

python3 scripts/model_pipline.py --index $(($SLURM_ARRAY_TASK_ID - 1))
