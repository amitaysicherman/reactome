import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
# from tqdm.notebook import tqdm

from biopax_parser import Reaction, reaction_from_str
from dataset_builder import get_reaction_entities
from dataset_builder import have_unkown_nodes, have_dna_nodes
from index_manger import NodesIndexManager, NodeTypes, PRETRAINED_EMD_FUSE, NodeData

DEBUG = True


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


def replace_molecule_protein(nodes, node_index_manager: NodesIndexManager, protein_or_mol=0):
    proteins = [node for node in nodes if node.type == NodeTypes.protein]
    molecules = [node for node in nodes if node.type == NodeTypes.molecule]
    if protein_or_mol == 0:
        if len(proteins) == 0:
            return None
        random_index = random.randint(0, len(proteins) - 1)
        replace_id = node_index_manager.sample_entity(proteins[random_index].index, how='random', what='protein')
        proteins[random_index] = node_index_manager.index_to_node[replace_id]

    else:
        if len(molecules) == 0:
            return None
        random_index = random.randint(0, len(molecules) - 1)
        replace_id = node_index_manager.sample_entity(molecules[random_index].index, how='random', what='molecule')
        molecules[random_index] = node_index_manager.index_to_node[replace_id]
    return proteins + molecules


def get_reaction_score(nodes: List[NodeData], how="mean"):
    scores = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            vec1 = nodes[i].vec
            vec2 = nodes[j].vec
            if (vec1 == 0).all() or (vec2 == 0).all():
                continue
            score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            scores.append(score)
    if len(scores) == 0:
        print("No scores")
        return 0.5
    if how == "min":
        return np.min(scores)
    if how == "mean":
        return np.mean(scores)
    if how == "median":
        return np.median(scores)


def get_replace_dataset(reactions: List[List[NodeData]], protein_or_mol):
    data = []
    labels = []
    for nodes in tqdm(reactions):
        if len(nodes) < 2:
            continue
        replace_prot = replace_molecule_protein(nodes, node_index_manager, protein_or_mol=protein_or_mol)
        if replace_prot is not None:
            data.append(nodes)
            labels.append(1)
            data.append(replace_prot)
            labels.append(0)
    return data, labels


def clean_reaction(reactions: List[Reaction], node_index_manager: NodesIndexManager):
    reactions = [reaction for reaction in reactions if
                 not have_unkown_nodes(reaction, node_index_manager, check_output=True)]
    reactions = [reaction for reaction in reactions if
                 not have_dna_nodes(reaction, node_index_manager, check_output=True)]
    return reactions


if __name__ == '__main__':
    node_index_manager = NodesIndexManager(fuse_vec=PRETRAINED_EMD_FUSE, fuse_config="8192_1_1024_0.0_0.001_1_512")
    root = "data/items"
    with open(f'{root}/reaction.txt') as f:
        lines = f.readlines()
    lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
    reactions = [reaction_from_str(line) for line in lines]

    types = [get_reaction_type(get_reaction_nodes(reaction, node_index_manager)) for reaction in
             clean_reaction(reactions, node_index_manager)]
    print("Reactions types:")
    print({t: types.count(t) for t in set(types)})

    train_reactions = clean_reaction(reactions[:int(len(reactions) * 0.8)], node_index_manager)
    train_reactions = [get_reaction_nodes(reaction, node_index_manager) for reaction in train_reactions]
    test_reactions = clean_reaction(reactions[int(len(reactions) * 0.8):], node_index_manager)
    test_reactions = [get_reaction_nodes(reaction, node_index_manager) for reaction in test_reactions]

    configs = [("protein", "protein"), ("molecule", "molecule"), ("protein", "both"), ("molecule", "both")]
    for replace_data, reaction_type in configs:
        train_config = [reaction for reaction in train_reactions if
                        reaction_type in get_reaction_type(reaction)]
        test_config = [reaction for reaction in test_reactions if reaction_type in get_reaction_type(reaction)]

        train_x, train_y = get_replace_dataset(train_config, replace_data == "molecule")
        test_x, test_y = get_replace_dataset(test_config, replace_data == "molecule")
        train_scores = [get_reaction_score(x) for x in tqdm(train_x)]
        test_scores = [get_reaction_score(x) for x in tqdm(test_x)]
        train_auc = roc_auc_score(train_y, train_scores)
        test_auc = roc_auc_score(test_y, test_scores)
        print(f"Replace {replace_data} in {reaction_type} reactions:")
        print(f"Train AUC: {train_auc}")
        print(f"Test AUC: {test_auc}")
