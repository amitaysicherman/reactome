#!/bin/sh
#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-60
#SBATCH --mem=32G
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/contrastive_learning.txt)
python contrastive_learning/trainer.py $args