#!/bin/sh
#SBATCH --time=0-3
#SBATCH --array=1-7
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_prot_drug.txt)
python protein_drug/train_eval.py $args --random_seed 42
python protein_drug/train_eval.py $args --random_seed 43
python protein_drug/train_eval.py $args --random_seed 44
python protein_drug/train_eval.py $args --random_seed 45
python protein_drug/train_eval.py $args --random_seed 46
python protein_drug/train_eval.py $args --random_seed 47
python protein_drug/train_eval.py $args --random_seed 48
python protein_drug/train_eval.py $args --random_seed 49
python protein_drug/train_eval.py $args --random_seed 50
python protein_drug/train_eval.py $args --random_seed 51


