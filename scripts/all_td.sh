#!/bin/sh
#!/bin/sh
#SBATCH --time=3-00
#SBATCH --array=1-106
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH --requeue

args=$(sed -n "$SLURM_ARRAY_TASK_ID"p scripts/all_td.txt)
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 0 --cafa_use_model 1

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 0 --cafa_use_model 1

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 0 --cafa_use_model 1

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 0 --cafa_use_model 1

python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 1
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 1 --cafa_use_model 0
python torchdrug_tasks/trainer.py $args --cafa_use_fuse 0 --cafa_use_model 1