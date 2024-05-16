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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(learned_embedding_dim=128, hidden_channels=128, num_layers=3, root="../data/items",
               layer_type="SAGEConv", return_reaction_embedding=False, model_path="../data/model/model.pt",out_channels=1,fuse=False):
    nodes_index_manager = NodesIndexManager(root,fuse_vec=fuse)

    model = HeteroGNN(nodes_index_manager, hidden_channels=hidden_channels, out_channels=out_channels,
                      num_layers=num_layers,
                      learned_embedding_dim=learned_embedding_dim, train_all_emd=False, save_activation=True,
                      conv_type=layer_type, return_reaction_embedding=return_reaction_embedding).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, nodes_index_manager


def reaction_to_data(reaction: Reaction, nodes_index_manager: NodesIndexManager):
    g, tags = get_nx_for_tags_prediction_task(reaction, nodes_index_manager)
    return nx_to_torch_geometric(g, tags=torch.Tensor(dataclasses.astuple(tags)).to(torch.float32))
