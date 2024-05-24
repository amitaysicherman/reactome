import os.path
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from biopax_parser import Reaction, reaction_from_str
from dataset_builder import build_dataset_for_tags_prediction_task, replace_entity_augmentation
from dataset_builder import get_reaction_entities
from dataset_builder import have_unkown_nodes, have_dna_nodes
from index_manger import NodesIndexManager, NodeTypes, NodeData
from model_tags import HeteroGNN

# from tqdm.notebook import tqdm

FUSE_COLUMNS = ['fuse_bs', 'fuse_do', 'fuse_h_dim', 'fuse_lr', 'fuse_nl', 'fuse_o_dim', 'fuse_mp', 'fuse_recon']
MODEL_COLUMNS = ['hidden_channels', 'layer_type', 'learned_embedding_dim', 'lr', 'num_layers', 'pretrained_method',
                 'sample', 'train_all_emd']
RESULTS_COLUMNS = ['protein_protein', 'molecule_molecule', 'protein_both', 'molecule_both']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = "data/items"
layer_type = "SAGEConv"


def get_args_from_name(name):
    name = name.replace("model_fake_fake_task_1-", "")
    name = "_".join(name.split("_")[:-1])  # remove the epoch number
    args = dict()
    for key_values in name.split("-"):
        if key_values.startswith("fuse_config"):
            key = "fuse_config"
            value = key_values.replace("fuse_config_", "")
            args[key] = value
            continue

        *key, value = key_values.split("_")
        key = "_".join(key)
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                pass

        args[key] = value
    print(args)
    return args


def get_reaction_type(nodes: List[NodeData]):
    proteins = [node for node in nodes if node.type == NodeTypes.protein]
    molecules = [node for node in nodes if node.type == NodeTypes.molecule]
    if len(proteins) > 0 and len(molecules) > 0:
        return "both"
    if len(proteins) > 0:
        return "protein"
    if len(molecules) > 0:
        return "molecule"
    return "none"


def get_reaction_nodes(reaction: Reaction, node_index_manager: NodesIndexManager):
    entities = get_reaction_entities(reaction, False)
    nodes = [node_index_manager.name_to_node[entity.get_unique_id()] for entity in entities]
    return nodes


def clean_reaction(reactions: List[Reaction], node_index_manager: NodesIndexManager):
    reactions = [reaction for reaction in reactions if
                 not have_unkown_nodes(reaction, node_index_manager, check_output=True)]
    reactions = [reaction for reaction in reactions if
                 not have_dna_nodes(reaction, node_index_manager, check_output=True)]
    return reactions


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_all_model_names():
    import os
    all_models = {model_name.replace(".pt", "") for model_name in os.listdir("data/model")}
    all_models = {model_name for model_name in all_models if model_name.startswith("model")}
    all_models_names = {"_".join(m.split("_")[:-1]) for m in all_models}

    last_epoch_models = []
    for prefix_model_name in all_models_names:
        all_options = [m for m in all_models if m.startswith(prefix_model_name)]
        epoch_to_model = {int(m.split("_")[-1]): m for m in all_options}
        model_name = epoch_to_model[max(epoch_to_model.keys())]
        last_epoch_models.append(model_name)
    return last_epoch_models


def get_model(model_name):
    args = get_args_from_name(model_name)
    print(args)
    node_index_manager = NodesIndexManager(root=root, fuse_config=args['fuse_config'],
                                           fuse_vec=args['pretrained_method'])
    hidden_channels = args['hidden_channels']
    num_layers = args['num_layers']
    learned_embedding_dim = args['learned_embedding_dim']
    train_all_emd = args['train_all_emd']

    model = HeteroGNN(node_index_manager, hidden_channels=hidden_channels, out_channels=1, num_layers=num_layers,
                      learned_embedding_dim=learned_embedding_dim, train_all_emd=train_all_emd, save_activation=False,
                      conv_type=layer_type).to(device)

    model.load_state_dict(torch.load(f"data/model/{model_name}.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, node_index_manager, args


def create_datasets(lines, node_index_manager: NodesIndexManager):
    datasets = {"protein_protein": [], "molecule_molecule": [], "protein_both": [], "molecule_both": [], }

    skip_count = 0
    for line in lines:
        reaction = reaction_from_str(line)
        reaction_type = get_reaction_type(get_reaction_nodes(reaction, node_index_manager))
        data = build_dataset_for_tags_prediction_task(line, node_index_manager, False)
        if data is None:
            continue
        if reaction_type == "protein":
            change_protein_data = replace_entity_augmentation(node_index_manager, data, NodeTypes.protein, "random")
            if change_protein_data is not None:
                datasets["protein_protein"].append(data)
                datasets["protein_protein"].append(change_protein_data)
        elif reaction_type == "molecule":
            change_molecule_data = replace_entity_augmentation(node_index_manager, data, NodeTypes.molecule, "random")
            if change_molecule_data is not None:
                datasets["molecule_molecule"].append(data)
                datasets["molecule_molecule"].append(change_molecule_data)
        elif reaction_type == "both":
            change_protein_data = replace_entity_augmentation(node_index_manager, data, NodeTypes.protein, "random")
            change_molecule_data = replace_entity_augmentation(node_index_manager, data, NodeTypes.molecule, "random")
            if change_protein_data is not None:
                datasets["protein_both"].append(data)
                datasets["protein_both"].append(change_protein_data)
            if change_molecule_data is not None:
                datasets["molecule_both"].append(data)
                datasets["molecule_both"].append(change_molecule_data)
        else:
            skip_count += 1
            continue
    return datasets


def apply_and_get_score(datasets, model, results):
    for name, all_data in datasets.items():
        preds = []
        real = []
        for data in all_data:
            data = data.to(device)
            with torch.no_grad():
                x_dict = {key: data.x_dict[key].to(device) for key in data.x_dict.keys()}
                y = data['tags'].to(device)
                y = y.float()
                y = y[-1:]
                edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
                out = model(x_dict, edge_index_dict)
                preds.extend(sigmoid(out.detach().cpu().numpy()).tolist())
                real.extend(y.detach().cpu().numpy().tolist())
        results[name].append(roc_auc_score(real, preds))
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_1024_0_0-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_2-sample_0-train_all_emd_0_1")
    parser.add_argument("--n", type=int, default=10)
    parser = parser.parse_args()
    results_file = "data/scores/scores.csv"
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            results_mean_values = [f"{key}_mean" for key in RESULTS_COLUMNS]
            results_std_values = [f"{key}_std" for key in RESULTS_COLUMNS]
            f.write(",".join(FUSE_COLUMNS + MODEL_COLUMNS + results_mean_values + results_std_values) + "\n")

    with open(f'{root}/reaction.txt') as f:
        lines = f.readlines()
    lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
    lines = lines[int(0.8 * len(lines)):]  # only test data

    if parser.model_name == "all":
        model_names = get_all_model_names()
    else:
        model_names = [parser.model_name]
    print(f"Model names: {model_names}")
    for model_name in model_names:
        model, node_index_manager, args = get_model(model_name)
        results = {"protein_protein": [], "molecule_molecule": [], "protein_both": [], "molecule_both": []}
        for _ in tqdm(range(parser.n)):
            datasets = create_datasets(lines, node_index_manager)
            results = apply_and_get_score(datasets, model, results)
        for key, values in results.items():
            print(f"{key}:{np.mean(values):.3f} +- {np.std(values):.3f}")
        fuse_values = args['fuse_config'].split("_")
        model_values = [args[key] for key in MODEL_COLUMNS]
        results_mean_values = [np.mean(results[key]) for key in RESULTS_COLUMNS]
        results_std_values = [np.std(results[key]) for key in RESULTS_COLUMNS]
        results = [*fuse_values, *model_values, *results_mean_values, *results_std_values]
        with open(results_file, "a") as f:
            f.write(",".join([str(val) for val in results]) + "\n")
