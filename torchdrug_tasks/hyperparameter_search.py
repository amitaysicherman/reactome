# hyperparameter_search.py

import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.stopper import TrialPlateauStopper
from trainer import train_model_with_config
from common.args_manager import get_args

def main(args):
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        'use_fuse': tune.choice([True, False]),
        'use_model': tune.choice([True, False]),
    }

    bayesopt_search = BayesOptSearch(metric="best_valid_score", mode="max")

    # Implement early stopping
    stopper = TrialPlateauStopper(
        metric="best_valid_score",
        mode="max",
        num_results=10,  # Number of trials without improvement before stopping
        grace_period=5,  # Minimum number of results before early stopping is considered
        patience=5,      # Number of results to wait for improvement
    )

    tune.run(
        tune.with_parameters(train_model_with_config, args=args),
        config=search_space,
        search_alg=bayesopt_search,
        num_samples=100,  # Set a higher number for broader exploration
        stop=stopper,  # Add the early stopping policy
    )


if __name__ == '__main__':
    main(get_args())
