import torch
import subprocess
from multiprocessing import Pool
import argparse

skip_names = []


def get_default_args():
    return {
        'name': 'default',
        'fuse_name': '',
        'fuse_recon': 0,
        'gnn_train_all_emd': 0,
        'gnn_hidden_channels': 256,
        'fuse_all_to_one': '',
        'gnn_pretrained_method': 1,
        # 'data_aug': 'all',
        # 'gnn_last_or_concat': 0,
        # 'gnn_reaction_or_mean': 0,
    }


def fill_node_emd_args(args, node_emd):
    if node_emd == "no":
        args['gnn_pretrained_method'] = 0
        args['gnn_train_all_emd'] = 1
        args['fuse_name'] = "no"
    elif node_emd == "pre":
        args['gnn_pretrained_method'] = 1
        args['fuse_name'] = "no"
    else:
        args['gnn_pretrained_method'] = 2
        if node_emd == "fuse":
            args['fuse_name'] = "fuse"
        elif node_emd == "recon":
            args['fuse_recon'] = 1
            args['fuse_name'] = "recon"
        elif node_emd == "all-to-prot":
            args['fuse_all_to_one'] = 'protein'
            args['fuse_name'] = "all-to-prot"
        elif node_emd == "all-to-mol":
            args['fuse_all_to_one'] = 'molecule'
            args['fuse_name'] = "all-to-mol"
        elif node_emd == "all-to-all":
            args['fuse_all_to_one'] = 'all'
            args['fuse_name'] = "all-to-all"


def fill_size_args(args, model_size):
    if model_size == "s":
        args['gnn_hidden_channels'] = 32
    elif model_size == "m":
        args['gnn_hidden_channels'] = 256
    else:
        args['gnn_hidden_channels'] = 512


def fill_graph_emb_args(args, graph_emb):
    if graph_emb == "reaction":
        args['gnn_reaction_or_mean'] = 0
    elif graph_emb == "mean":
        args['gnn_reaction_or_mean'] = 1
    elif graph_emb == "concat":
        args['gnn_last_or_concat'] = 1
    else:
        args['gnn_last_or_concat'] = 1
        args['gnn_reaction_or_mean'] = 1


def fill_aug_data_args(args, aug_data):
    args['data_aug'] = aug_data


def args_to_str(args):
    args_str = ""
    for key, value in args.items():
        if value != "":
            args_str += f"--{key} {value} "
    return args_str.strip()


def get_args(node_emd, model_size, graph_emb, aug_data):
    args = get_default_args()
    fill_node_emd_args(args, node_emd)
    fill_size_args(args, model_size)
    # fill_graph_emb_args(args, graph_emb)
    # fill_aug_data_args(args, aug_data)
    name = f"{node_emd}_{model_size}"#_{graph_emb}_{aug_data}"
    args['name'] = name
    return args_to_str(args), name


num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
max_concurrent_runs = 48
counter = 0


def run_cmd(cmd):
    subprocess.run(cmd, shell=True)


def run_commands(commands):
    with Pool(max_concurrent_runs) as p:
        p.map(run_cmd, commands)


commands = []
for i, name in enumerate(["no", "fuse", "recon", "all-to-prot", "all-to-all"]):
    args = get_default_args()
    fill_node_emd_args(args, name)
    args['name'] = name
    gpu_index = i % num_gpus
    print(args_to_str(args))
    script = f"python3 model/contrastive_learning.py {args_to_str(args)}"
    cmd = f'CUDA_VISIBLE_DEVICES="{gpu_index}" bash -c "{script}"'
    commands.append(cmd)
# run_commands(commands)

commands = []

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=-1)
args = parser.parse_args()
node_emd_list = ["recon", "all-to-prot", "all-to-all", "no", "pre", "fuse"]
if args.index != -1:
    node_emd_list = [node_emd_list[args.index]]
for node_emd in node_emd_list:
    for model_size in ["s", "m", "l"]:
        # for aug_data in ["all"]:
            # for graph_emb in ["reaction"]:#, "mean", "concat", "both"]:
                graph_emb, aug_data = "reaction", "all"

                args, name = get_args(node_emd, model_size, graph_emb, aug_data)
                print(args)
                if name in skip_names:
                    continue
                rm_cmd = f'rm -rf data/models_checkpoints/gnn_{name}'
                script = f"python3 model/train_gnn.py {args} && python model/eval_model.py {args} && {rm_cmd}"
                counter += 1
                gpu_index = counter % num_gpus
                cmd = f'CUDA_VISIBLE_DEVICES="{gpu_index}" bash -c "{script}"'
                commands.append(cmd)
# print(commands)
# run_commands(commands)

print(counter)
