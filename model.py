import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, to_hetero
from torch_geometric.nn import DataParallel

from nodes_indexes import NodesIndexManager, get_types_values, get_edges_values
from reaction_to_graph import ReactionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nodes = ["reaction", "entities", "locations", "catalysis", "catalyst_activities", "modifications"]


class PartialFixedEmbedding(nn.Module):
    def __init__(self, embedding_dim, index_to_value: dict):
        super(PartialFixedEmbedding, self).__init__()
        self.embeddings = dict()

        for index, value in index_to_value.items():
            if value is not None:
                self.embeddings[index] = nn.Parameter(torch.tensor(value, dtype=torch.float32).to(device))
            else:
                self.embeddings[index] = nn.Parameter(nn.init.uniform_(torch.zeros(embedding_dim), -1.0, 1.0).to(device))

    def forward(self, input):
        input = input.flatten()
        input = torch.stack([self.embeddings[i.item()] for i in input])
        return input

    def print_means(self):
        have_t = 10
        have_f = 10
        for key, value in self.embeddings.items():
            if have_f == 0 and have_t == 0:
                break
            if value.requires_grad:
                if have_t == 0:
                    continue
                have_t -= 1
            else:
                if have_f == 0:
                    continue
                have_f -= 1

            print(key, value.requires_grad, value.sum())


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {e: SAGEConv(-1, hidden_channels).to(device) for e in get_edges_values()}, aggr='sum')
            self.convs.append(conv)

        self.lin_reaction = Linear(hidden_channels, out_channels).to(device)
        self.lin_nodes = Linear(hidden_channels, out_channels).to(device)

    def forward(self, x_dict, edge_index_dict, output_notes_opt):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # x_dict = {key: x.relu() for key, x in x_dict.items()}
        reaction_output = self.lin_reaction(x_dict['reaction'])
        output_notes_opt = self.lin_nodes(output_notes_opt)
        return (F.cosine_similarity(reaction_output, output_notes_opt) + 1) / 2


if __name__ == "__main__":
    dataset = ReactionDataset(sample=1, task="output_node")
    node_index_manager = NodesIndexManager(load_vectors=True)

    emb_model = PartialFixedEmbedding(1024, node_index_manager.vector).to(device)

    model = HeteroGNN(hidden_channels=1024, out_channels=64).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    nodes_types = get_types_values()
    edges_types = get_edges_values()
    metadata = (nodes_types, edges_types)

    for i in range(100):
        tot_loss = 0
        for data in dataset:
            x_dict = {key: emb_model(data.x_dict[key]) for key in data.x_dict.keys() if key not in ['y', 'output_vec']}
            y = data['y'].to(device)
            output_vec = torch.Tensor(data['output_vec']).to(device)
            out = model(x_dict, data.edge_index_dict, output_vec)
            loss = F.binary_cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print(i, tot_loss / len(dataset), out, y)
    print("Training completed!")
