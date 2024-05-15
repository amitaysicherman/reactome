from tqdm import tqdm
from matplotlib import pyplot as plt
from index_manger import NodesIndexManager, NodeTypes, node_colors
from biopax_parser import reaction_from_str, Reaction
import networkx as nx
from tagging import tag
import torch
from tagging import ReactionTag
from model_tags import HeteroGNN
import random
from dataset_builder import get_nx_for_tags_prediction_task, nx_to_torch_geometric, have_unkown_nodes, \
    replace_location_augmentation, replace_entity_augmentation, add_if_not_none
import dataclasses
import numpy as np
import seaborn as sns
import requests
from typing import List
from utils import load_model
from utils import reaction_to_data
from sklearn.metrics.pairwise import cosine_similarity

sns.set_theme(style="white")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NT = NodeTypes()
root = "../data/items"

nodes_index_manager = NodesIndexManager(root)

with open(f'{root}/reaction.txt') as f:
    lines = f.readlines()
lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
reactions = [reaction_from_str(line) for line in tqdm(lines)]
names = []
dataset = []
for reaction in reactions:
    if have_unkown_nodes(reaction, nodes_index_manager):
        continue
    data = reaction_to_data(reaction, nodes_index_manager)
    if data is None or data.tags.sum().item() == 0:
        continue
    dataset.append(data)
    names.append(reaction.reactome_id)

model, nodes_index_manager = load_model(learned_embedding_dim=256, hidden_channels=256, num_layers=3,out_channels=5,
    model_path="/home/amitay/PycharmProjects/reactome/data/model/model_mlc_256_2.pt", return_reaction_embedding=True)

all_emb = []
for i, data in enumerate(tqdm(dataset)):
    with torch.no_grad():
        _, emd = model(data.x_dict, data.edge_index_dict)
    all_emb.append(emd.cpu().numpy())
train_test_split = int(0.8 * len(all_emb))
train_emd = np.concatenate(all_emb[:train_test_split], axis=0)
test_emd = np.concatenate(all_emb[train_test_split:], axis=0)

similarity = cosine_similarity(test_emd, train_emd)
most_similar = np.argmax(similarity, axis=1)
most_similar_names = [names[i] for i in most_similar]
for i, (name, most_similar_name) in enumerate(zip(names[train_test_split:], most_similar_names)):
    print(f"{name} -> {most_similar_name}")
