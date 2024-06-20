import os
import os.path
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common.data_types import Reaction, NodeTypes
from common.path_manager import model_path, scores_path
from common.utils import reaction_from_str, sigmoid, get_best_gnn_cp
from dataset.dataset_builder import get_reaction_entities, get_reactions
from dataset.dataset_builder import have_unkown_nodes, have_dna_nodes
from dataset.dataset_builder import reaction_to_data, replace_entity_augmentation, replace_location_augmentation
from dataset.index_manger import NodesIndexManager, NodeData
from model.gnn_models import GnnModelConfig, HeteroGNN

RESULTS_COLUMNS = ['protein_protein', 'molecule_molecule', 'protein_both', 'molecule_both', 'location_both']

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


def get_all_model_names(aug_data):
    model_names = [x.replace("gnn_") for x in os.listdir(model_path) if x.startswith("gnn_")]
    best_models = [get_best_gnn_cp(name,aug_data) for name in model_names]
    return best_models


def get_model(cp_name, config_name) -> HeteroGNN:
    config = GnnModelConfig.load_from_file(config_name)
    model = HeteroGNN(config)
    model.load_state_dict(torch.load(cp_name, map_location=torch.device('cpu')))
    model.eval()
    return model


def create_datasets(lines, node_index_manager: NodesIndexManager):
    datasets = {x: [] for x in RESULTS_COLUMNS}

    skip_count = 0
    for reaction in lines:
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
            change_location_data = replace_location_augmentation(node_index_manager, data)
            if change_protein_data is not None:
                datasets["protein_both"].append(data)
                datasets["protein_both"].append(change_protein_data)
            if change_molecule_data is not None:
                datasets["molecule_both"].append(data)
                datasets["molecule_both"].append(change_molecule_data)
            if change_location_data is not None:
                datasets['location_both'].append(data)
                datasets['location_both'].append(change_location_data)
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
    from common.args_manager import get_args

    args = get_args()
    results_file = f"{scores_path}/summary_gnn.csv"

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            results_mean_values = [f"{key}_mean" for key in RESULTS_COLUMNS]
            results_std_values = [f"{key}_std" for key in RESULTS_COLUMNS]
            f.write(",".join(['name'] + results_mean_values + results_std_values) + "\n")

    _, _, lines = get_reactions()

    if args.name == "all":
        model_names = get_all_model_names(args.data_aug)
    else:
        model_names = [get_best_gnn_cp(args.name, args.data_aug)]
    print(f"Model names: {model_names}")
    for model_name in model_names:

        config_file = os.path.join(os.path.dirname(model_name), "config.txt")
        model = get_model(model_name, config_file)
        model = model.to(device)
        node_index_manager = model.emb.node_index_manager
        results = {x: [] for x in RESULTS_COLUMNS}
        for _ in tqdm(range(args.eval_n)):
            datasets = create_datasets(lines, node_index_manager)
            results = apply_and_get_score(datasets, model, results)
        for key, values in results.items():
            print(f"{key}:{np.mean(values):.3f} +- {np.std(values):.3f}")
        results_mean_values = [np.mean(results[key]) for key in RESULTS_COLUMNS]
        results_std_values = [np.std(results[key]) for key in RESULTS_COLUMNS]
        name = os.path.dirname(model_name).replace(model_path, "")
        name = name.replace("/", "")
        results = [name, *results_mean_values, *results_std_values]
        with open(results_file, "a") as f:
            f.write(",".join([str(val) for val in results]) + "\n")
