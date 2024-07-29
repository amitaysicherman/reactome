from tqdm import tqdm
import os
from trainer import train_model_with_config
from common.args_manager import get_args
from common.path_manager import scores_path
import pandas as pd

config_cols = ['bs', 'lr', 'use_fuse', 'use_model', 'n_layers', 'hidden_dim', 'drop_out']
n_max = 100


def main(args):
    search_space = {
        "bs": [8, 256, 1024],
        "lr": [5e-5, 5e-3],
        'use_fuse': [True],
        'use_model': [True, False],
        'n_layers': [1, 2],
        'hidden_dim': [256, 512],
        'drop_out': [0, 0.1, 0.5]
    }
    all_options = []
    for bs in search_space["bs"]:
        for lr in search_space["lr"]:
            for use_fuse in search_space["use_fuse"]:
                for use_model in search_space["use_model"]:
                    for n_layers in search_space["n_layers"]:
                        for hidden_dim in search_space["hidden_dim"]:
                            for drop_out in search_space["drop_out"]:
                                all_options.append({
                                    "bs": bs,
                                    "lr": lr,
                                    "use_fuse": use_fuse,
                                    "use_model": use_model,
                                    "n_layers": n_layers,
                                    "hidden_dim": hidden_dim,
                                    "drop_out": drop_out
                                })

    args = {
        "task_name": args.task_name,
        "fuse_base": args.dp_fuse_base,
        "mol_emd": args.mol_emd,
        "protein_emd": args.protein_emd,
        "print_output": False,
        "max_no_improve": args.max_no_improve,
        "fuse_model": None,
    }
    name = f'{args["task_name"]}_{args["protein_emd"]}_{args["mol_emd"]}'
    filename = f'{scores_path}/hp_{name}_torchdrug.csv'
    if os.path.exists(filename):
        os.remove(filename)
    all_cols = ["valid_score", "test_score"] + config_cols
    with open(filename, "w") as f:
        f.write(",".join(all_cols) + "\n")


    for option in tqdm(all_options[:1]):
        val_score, test_score = train_model_with_config(option, **args)
        values = [val_score, test_score] + [option.get(col, None) for col in config_cols]
        with open(filename, "a") as f:
            f.write(",".join(map(str, values)) + "\n")

    df = pd.read_csv(filename)
    df = df.sort_values("valid_score", ascending=False)
    best_config = {col: df.iloc[0][col] for col in config_cols}
    for col in ["bs", "n_layers", "hidden_dim"]:
        best_config[col] = int(best_config[col])
    for col in ["lr", "drop_out"]:
        best_config[col] = float(best_config[col])

    best_fuse_test_score = df.iloc[0]["test_score"]
    print(f"Best config: {best_config}")
    best_config['use_fuse'] = False
    best_config['use_model'] = True
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
