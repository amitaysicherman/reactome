import dataclasses

from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv, SAGEConv, Linear
import numpy as np

from common.scorer import Scorer
from dataset.index_manger import NodesIndexManager, PRETRAINED_EMD_FUSE
from common.utils import get_edges_values, args_to_str
from common.data_types import NodeTypes, DATA_TYPES, EMBEDDING_DATA_TYPES, REAL, FAKE_LOCATION_ALL, \
    FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE
from dataset.dataset_builder import get_data
from tagging import ReactionTag
from collections import defaultdict
from torch_geometric.loader import DataLoader
import seaborn as sns
from common.path_manager import scores_path, model_path
from model.gnn_models import GnnModelConfig, HeteroGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReactionBPHead(nn.Module):
    def __init__(self, n_layers, input_dim, output_dim, hidden_dim):

        super(ReactionBPHead, self).__init__()
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":

    import argparse
    from model.eval_model import get_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="model_conv_type_SAGEConv-epochs_10-fake_task_1-fuse_config_8192_1_1024_0.0_0.001_1_512-hidden_channels_256-learned_embedding_dim_256-lr_0.001-num_layers_3-out_channels_1-pretrained_method_1-return_reaction_embedding_0-sample_10-train_all_emd_0_0")
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=1024)

    parser = parser.parse_args()

    model, args = get_model(parser.model_name,True)
    model.eval()
    node_index_manager = model.emb.node_index_manager

    train_dataset, test_dataset, _ = get_data(node_index_manager, sample=0, location_augmentation_factor=0,
                                              entity_augmentation_factor=0,
                                              fake_task=1)

    reaction_bp_head = ReactionBPHead(parser.num_layers, args['hidden_channels'], len(node_index_manager.bp_name_to_index),
                                      parser.hidden_channels)
    reaction_bp_head.train()
    optimizer = torch.optim.Adam(reaction_bp_head.parameters(), lr=parser.lr)
    loss_func = nn.CrossEntropyLoss()

    for data in train_dataset:
        x_dict = {key: data.x_dict[key].to(device) for key in data.x_dict.keys()}
        y = data['bp'].to(device)
        y = y.long()
        if y.item() == -1:
            continue
        edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
        _, reaction_rep = model(x_dict, edge_index_dict)
        out = reaction_bp_head(reaction_rep)
        loss = loss_func(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
