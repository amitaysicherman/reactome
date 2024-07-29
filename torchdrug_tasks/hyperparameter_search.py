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

config_cols = ['bs', 'lr', 'use_fuse', 'use_model', 'n_layers', 'hidden_dim', 'drop_out']
n_max = 100


class CSVLoggerCallback(tune.Callback):
    def __init__(self, name):
        self.filename = f'{scores_path}/hp_{name}_torchdrug.csv'
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        all_cols = ["trial_id", "iteration", "valid_score", "test_score"] + config_cols
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                f.write(",".join(all_cols) + "\n")
        values = [trial.trial_id, iteration, result.get("valid_score", None), result.get("test_score", None)]
        values += [trial.config.get(col, None) for col in config_cols]
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
        max_t=n_max,  # Maximum number of epochs
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
    csv_logger = CSVLoggerCallback(name)

    tune.run(
        tune.with_parameters(train_model_with_config, **args),
        config=search_space,
        search_alg=optuna_search,  # Use OptunaSearch instead of BayesOptSearch
        scheduler=scheduler,  # Use ASHAScheduler
        num_samples=n_max,
        resources_per_trial={"cpu": 1, "gpu": 0},
        callbacks=[csv_logger],
    )

    df = pd.read_csv(csv_logger.filename)
    df = df.sort_values("valid_score", ascending=False)
    best_config = {col: df.iloc[0][col] for col in config_cols}
    best_fuse_test_score = df.iloc[0]["test_score"]
    print(f"Best config: {best_config}")
    best_config['use_fuse'] = False
    best_config['use_model'] = True
    args["tune_mode"] = False
    _, best_model_test_score = train_model_with_config(best_config, **args)

    header = ['task', 'mol_emd', 'protein_emd', 'score_model', 'score_fuse'] + config_cols
    output_file = f"{scores_path}/torchdrug.csv"
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(",".join(header) + "\n")
    values = [args["task_name"], args["mol_emd"], args["protein_emd"], best_model_test_score,
              best_fuse_test_score] + [best_config.get(col, None) for col in config_cols]
    with open(output_file, "a") as f:
        f.write(",".join(map(str, values)) + "\n")


if __name__ == '__main__':
    main(get_args())
