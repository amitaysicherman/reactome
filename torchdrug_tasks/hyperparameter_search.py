# hyperparameter_search.py

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import os
import torch
from trainer import train_model_with_config
from common.args_manager import get_args
from common.path_manager import scores_path
import pandas as pd


class CSVLoggerCallback(tune.Callback):
    def __init__(self, name):
        self.filename = f'{scores_path}/hp_{name}_torchdrug.csv'

    def on_trial_result(self, iteration, trials, trial, result, **info):
        config_cols = list(trial.config.keys())
        all_cols = ["trial_id", "iteration", "valid_score", "test_score"] + config_cols
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write(",".join(all_cols) + "\n")
        values = [trial.trial_id, iteration, result.get("valid_score", None), result.get("test_score", None)] + [
            trial.config[col] for col in config_cols]
        with open(self.filename, "a") as f:
            f.write(",".join(map(str, values)) + "\n")


def main(args):
    search_space = {
        "bs": tune.choice([8, 64, 256, 1024]),
        "lr": tune.loguniform(1e-5, 1e-2),
        'use_fuse': tune.choice([True]),
        'use_model': tune.choice([True, False]),
        'n_layers': tune.choice([1, 2, 3]),
        'hidden_dim': tune.choice([128, 256, 512, 1024]),
        'drop_out': tune.uniform(0.0, 0.5)
    }

    optuna_search = OptunaSearch(metric="valid_score", mode="max")

    # Use Async HyperBand with early stopping
    scheduler = ASHAScheduler(
        metric="valid_score",
        mode="max",
        max_t=250,  # Maximum number of epochs
        grace_period=5,  # Minimum epochs before stopping
        reduction_factor=2,  # Halving rate for early stopping
        brackets=1  # Number of brackets for successive halving
    )

    args = {
        "task_name": args.task_name,
        "fuse_base": args.dp_fuse_base,
        "mol_emd": args.mol_emd,
        "protein_emd": args.protein_emd,
        "print_output": False,
        "max_no_improve": args.max_no_improve,
        "fuse_model": None,
        "tune_mode": True
    }
    name = f'{args["task_name"]}_{args["protein_emd"]}_{args["mol_emd"]}'
    tune.run(
        tune.with_parameters(train_model_with_config, **args),
        config=search_space,
        search_alg=optuna_search,  # Use OptunaSearch instead of BayesOptSearch
        scheduler=scheduler,  # Use ASHAScheduler
        num_samples=50,
        resources_per_trial={"cpu": os.cpu_count(), "gpu": torch.cuda.device_count()},
        callbacks=[CSVLoggerCallback(name)],
    )


if __name__ == '__main__':
    main(get_args())
