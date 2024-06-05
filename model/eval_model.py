import os.path
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common.utils import reaction_from_str, get_last_epoch_model, sigmoid
from common.data_types import Reaction, NodeTypes
from dataset.dataset_builder import reaction_to_data, replace_entity_augmentation
from dataset.dataset_builder import get_reaction_entities
from dataset.dataset_builder import have_unkown_nodes, have_dna_nodes
from dataset.index_manger import NodesIndexManager, NodeData
from model.gnn_models import GnnModelConfig, HeteroGNN
from common.path_manager import reactions_file, model_path, scores_path
import os

RESULTS_COLUMNS = ['protein_protein', 'molecule_molecule', 'protein_both', 'molecule_both']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    nodes = [node_index_manager.name_to_node[entity.get_db_identifier()] for entity in entities]
    return nodes


def clean_reaction(reactions: List[Reaction], node_index_manager: NodesIndexManager):
    reactions = [reaction for reaction in reactions if
                 not have_unkown_nodes(reaction, node_index_manager, check_output=True)]
    reactions = [reaction for reaction in reactions if
                 not have_dna_nodes(reaction, node_index_manager, check_output=True)]
    return reactions


def get_all_model_names(cp_idx: int):
    model_names = [x for x in os.listdir(model_path) if x.startswith("gnn_")]
    last_epoch_models = [get_last_epoch_model(f"{model_path}/{model_dir}/", cp_idx) for model_dir in model_names]
    return last_epoch_models


def get_model(cp_name, config_name) -> HeteroGNN:
    config = GnnModelConfig.load_from_file(config_name)
    model = HeteroGNN(config)
    model.load_state_dict(torch.load(cp_name, map_location=torch.device('cpu')))
    model.eval()
    return model


def create_datasets(lines, node_index_manager: NodesIndexManager):
    datasets = {"protein_protein": [], "molecule_molecule": [], "protein_both": [], "molecule_both": []}

    skip_count = 0
    for line in lines:
        reaction = reaction_from_str(line)
        reaction_type = get_reaction_type(get_reaction_nodes(reaction, node_index_manager))
        data = reaction_to_data(line, node_index_manager, True)
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
                edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
                out = model(x_dict, edge_index_dict)
                preds.append(sigmoid(out.detach().cpu().numpy()).tolist()[0][0])
                real.append(y.item())
        results[name].append(roc_auc_score(real, preds))
        print(name, results[name][-1])
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gnn_default")
    parser.add_argument("--cp_idx", type=int, default=-1)
    parser.add_argument("--n", type=int, default=10)

    parser = parser.parse_args()
    results_file = f"{scores_path}/summary_gnn.csv"

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            results_mean_values = [f"{key}_mean" for key in RESULTS_COLUMNS]
            results_std_values = [f"{key}_std" for key in RESULTS_COLUMNS]
            f.write(",".join(['name'] + results_mean_values + results_std_values) + "\n")

    with open(reactions_file) as f:
        lines = f.readlines()
    lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
    lines = lines[int(0.8 * len(lines)):]  # only test data

    if parser.model_name == "all":
        model_names = get_all_model_names(parser.cp_idx)
    else:
        model_names = [get_last_epoch_model(f"{model_path}/{parser.model_name}", parser.cp_idx)]
    print(f"Model names: {model_names}")
    for model_name in model_names:

        config_file = os.path.join(os.path.dirname(model_name), "config.txt")
        model = get_model(model_name, config_file)
        model = model.to(device)
        node_index_manager = model.emb.node_index_manager
        results = {"protein_protein": [], "molecule_molecule": [], "protein_both": [], "molecule_both": []}
        for _ in tqdm(range(parser.n)):
            datasets = create_datasets(lines, node_index_manager)
            results = apply_and_get_score(datasets, model, results)
        for key, values in results.items():
            print(f"{key}:{np.mean(values):.3f} +- {np.std(values):.3f}")
        results_mean_values = [np.mean(results[key]) for key in RESULTS_COLUMNS]
        results_std_values = [np.std(results[key]) for key in RESULTS_COLUMNS]
        name = os.path.dirname(model_name).replace(model_path, "")
        name = name.replace("/", "")
        if parser.cp_idx != -1:
            name += f"_{parser.cp_idx}"
        results = [name, *results_mean_values, *results_std_values]
        with open(results_file, "a") as f:
            f.write(",".join([str(val) for val in results]) + "\n")
