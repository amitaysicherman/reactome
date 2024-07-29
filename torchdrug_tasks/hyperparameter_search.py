# hyperparameter_search.py

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from trainer import train_model_with_config
from common.args_manager import get_args


def main(args):
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        'use_fuse': tune.choice([True, False]),
        'use_model': tune.choice([True, False]),
    }

    optuna_search = OptunaSearch(metric="best_valid_score", mode="max")

    # Use Async HyperBand with early stopping
    scheduler = ASHAScheduler(
        metric="best_valid_score",
        mode="max",
        max_t=250,  # Maximum number of epochs
        grace_period=5,  # Minimum epochs before stopping
        reduction_factor=2,  # Halving rate for early stopping
        brackets=1  # Number of brackets for successive halving
    )

    config = {
        "use_fuse": args.cafa_use_fuse,
        "use_model": args.cafa_use_model,
        "bs": args.dp_bs,
        "lr": args.dp_lr
    }
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
        tune.with_parameters(train_model_with_config(), **args),
        config=search_space,
        search_alg=optuna_search,  # Use OptunaSearch instead of BayesOptSearch
        scheduler=scheduler,  # Use ASHAScheduler
        num_samples=50,
    )


if __name__ == '__main__':
    main(get_args())
