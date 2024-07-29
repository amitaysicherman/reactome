import subprocess


def submit_slurm_job(command, i):
    """Submits a Slurm job using sbatch --wrap"""
    slurm_command = f"sbatch --job-name td-{i} --time=1-00 --mem=64G --requeue --cpus-per-task=8 --wrap 'python torchdrug_tasks/hyperparameter_search.py {command}'"
    subprocess.run(slurm_command, shell=True, check=True)


# Read commands from the text file
with open("scripts/all_td.txt", "r") as f:
    commands = f.readlines()

# Submit Slurm jobs for each command
for i, command in enumerate(commands):
    command = command.strip()  # Remove leading/trailing whitespace
    submit_slurm_job(command, i)
