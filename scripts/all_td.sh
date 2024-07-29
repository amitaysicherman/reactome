#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-19
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_td.txt)
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args
python torchdrug_tasks/trainer.py $args