#!/bin/sh
#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-256
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_td.txt)
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0