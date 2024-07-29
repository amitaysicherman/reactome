# hyperparameter_search.py

from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from trainer import train_model_with_config
from common.args_manager import get_args


def main(args):
    search_space = {
        "batch_size": tune.choice([8, 64, 256, 1024]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        'use_fuse': tune.choice([True, False]),
        'use_model': tune.choice([True, False]),
    }

    bayesopt_search = BayesOptSearch(metric="validation_score", mode="max")

    tune.run(
        tune.with_parameters(train_model_with_config, args=args),
        config=search_space,
        search_alg=bayesopt_search,
        num_samples=20,  # Number of trials to run
        resources_per_trial={"cpu": 2, "gpu": 1}  # Adjust based on your available resources
    )


if __name__ == '__main__':
    main(get_args())
