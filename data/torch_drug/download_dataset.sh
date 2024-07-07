#!/bin/sh
#SBATCH --time=0-12
#SBATCH --array=1-3
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

python3 data/torch_drug/download_dataset.py --index $SLURM_ARRAY_TASK_ID