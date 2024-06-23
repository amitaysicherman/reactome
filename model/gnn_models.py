import dataclasses
from collections import defaultdict

import numpy as np
import torch
from torch import nn as nn

from torch_geometric.nn import TransformerConv, SAGEConv, HeteroConv, Linear

from common.data_types import NodeTypes, DATA_TYPES, EMBEDDING_DATA_TYPES, EdgeTypes, REACTION
from dataset.index_manger import NodesIndexManager


@dataclasses.dataclass
class GnnModelConfig:
    learned_embedding_dim: int
    hidden_channels: int
    num_layers: int
    conv_type: str
    train_all_emd: int
    fake_task: int
    pretrained_method: int
    fuse_name: str
    out_channels: int
    last_or_concat: int
    fuse_pretrained_start: int
    # reaction_or_mean: int
    drop_out: float

    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            for k, v in dataclasses.asdict(self).items():
                f.write(f"{k}={v}\n")

    @staticmethod
    def load_from_file(file_name):
        with open(file_name) as f:
            data = {}
            for line in f:
                k, v = line.strip().split("=")
                data[k] = v
        return GnnModelConfig(learned_embedding_dim=int(data["learned_embedding_dim"]),
                              hidden_channels=int(data["hidden_channels"]),
                              num_layers=int(data["num_layers"]),
                              conv_type=data["conv_type"],
                              train_all_emd=int(data["train_all_emd"]),
                              fake_task=int(data["fake_task"]),
                              pretrained_method=int(data["pretrained_method"]),
                              fuse_name=data["fuse_name"],
                              out_channels=int(data["out_channels"]),
                              last_or_concat=int(data["last_or_concat"]),
                              # reaction_or_mean=int(data["reaction_or_mean"]),
                              fuse_pretrained_start=int(data["fuse_pretrained_start"]),
                              drop_out=float(data["drop_out"])
                              )


class PartialFixedEmbedding(nn.Module):
    def __init__(self, node_index_manager: NodesIndexManager, learned_embedding_dim, train_all_emd):
        super(PartialFixedEmbedding, self).__init__()
        self.node_index_manager = node_index_manager
        self.embeddings = nn.ModuleDict()
        self.embeddings[NodeTypes.reaction] = nn.Embedding(1, learned_embedding_dim)
        self.embeddings[NodeTypes.complex] = nn.Embedding(1, learned_embedding_dim)
        for dtype in DATA_TYPES:
            if dtype in EMBEDDING_DATA_TYPES:
                dtype_indexes = range(node_index_manager.dtype_to_first_index[dtype],
                                      node_index_manager.dtype_to_last_index[dtype])
                vectors = np.stack([node_index_manager.index_to_node[i].vec for i in dtype_indexes])
                self.embeddings[dtype] = nn.Embedding.from_pretrained(
                    torch.tensor(vectors, dtype=torch.float32))
                if not train_all_emd:
                    self.embeddings[dtype].requires_grad_(False)
                else:
                    self.embeddings[dtype].requires_grad_(True)
            else:
                n_samples = node_index_manager.dtype_to_last_index[dtype] - node_index_manager.dtype_to_first_index[
                    dtype]
                self.embeddings[dtype] = nn.Embedding(n_samples, learned_embedding_dim)

    def forward(self, x):
        x = x.squeeze(1)
        dtype = self.node_index_manager.index_to_node[x[0].item()].type
        if dtype == NodeTypes.reaction or dtype == NodeTypes.complex:
            return self.embeddings[dtype](torch.zeros_like(x))
        x = x - self.node_index_manager.dtype_to_first_index[dtype]
        x = self.embeddings[dtype](x)
        return x


conv_type_to_class = {
    "TransformerConv": TransformerConv,
    "SAGEConv": SAGEConv
}
conv_type_to_args = {
    "TransformerConv": {},
    "SAGEConv": {"normalize": True}
}


def get_edges_values():
    attributes = dir(EdgeTypes)
    edges = []
    for attr in attributes:
        value = getattr(EdgeTypes, attr)
        if isinstance(value, tuple) and len(value) == 3:
            edges.append(value)
    return edges


class HeteroGNN(torch.nn.Module):
    def __init__(self, config: GnnModelConfig):

        super().__init__()
        node_index_manager = NodesIndexManager(pretrained_method=config.pretrained_method, fuse_name=config.fuse_name,
                                               fuse_pretrained_start=config.fuse_pretrained_start)

        self.emb = PartialFixedEmbedding(node_index_manager, config.learned_embedding_dim, config.train_all_emd)
        self.convs = torch.nn.ModuleList()
        self.save_activations = defaultdict(list)
        self.last_or_concat = config.last_or_concat
        # self.reaction_or_mean = config.reaction_or_mean
        self.drop_out = nn.Dropout(config.drop_out)
        for i in range(config.num_layers):
            conv_per_edge = {}

            for edge in get_edges_values():
                e_conv = conv_type_to_class[config.conv_type](-1, config.hidden_channels,
                                                              **conv_type_to_args[config.conv_type])
                conv_per_edge[edge] = e_conv
            conv = HeteroConv(conv_per_edge, aggr='sum')
            self.convs.append(conv)
        if self.last_or_concat:
            self.lin_reaction = Linear(config.hidden_channels * config.num_layers, config.out_channels)
        else:
            self.lin_reaction = Linear(config.hidden_channels, config.out_channels)

    def forward(self, x_dict, edge_index_dict, return_reaction_embedding=False):
        for key, x in x_dict.items():
            x_dict[key] = self.emb(x)
        all_emb = []
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.drop_out(x) for key, x in x_dict.items()}

            # if self.reaction_or_mean:
            #     nodes = []
            #     for dtype in [NodeTypes.complex, NodeTypes.molecule, NodeTypes.protein, NodeTypes.dna]:
            #         if dtype in x_dict:
            #             nodes.append(x_dict[dtype])
            #     nodes = torch.cat(nodes, dim=0)
            #     nodes = nodes.mean(dim=0).unsqueeze(0)

            # else:
            nodes = x_dict[REACTION]

            all_emb.append(nodes)
        if self.last_or_concat:
            all_emb = torch.cat(all_emb, dim=-1)
        else:
            all_emb = all_emb[-1]
        prediction = self.lin_reaction(all_emb)
        if return_reaction_embedding:
            return prediction, all_emb
        return prediction
