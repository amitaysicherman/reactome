# hyperparameter_search.py

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import os
import torch
from trainer import train_model_with_config
from common.args_manager import get_args


def main(args):
    search_space = {
        "bs": tune.choice([8, 64, 256, 1024]),
        "lr": tune.loguniform(1e-5, 1e-2),
        'use_fuse': tune.choice([True]),
        'use_model': tune.choice([True, False]),
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
        "print_output": args.dp_print,
        "max_no_improve": args.max_no_improve,
        "fuse_model": None,
        "tune_mode": True
    }

    tune.run(
        tune.with_parameters(train_model_with_config, **args),
        config=search_space,
        search_alg=optuna_search,  # Use OptunaSearch instead of BayesOptSearch
        scheduler=scheduler,  # Use ASHAScheduler
        num_samples=50,
        resources_per_trial={"cpu": os.cpu_count(), "gpu": torch.cuda.device_count()},

    )


if __name__ == '__main__':
    main(get_args())
