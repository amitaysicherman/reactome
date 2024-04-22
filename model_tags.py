import dataclasses
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.nn import DataParallel

from index_manger import NodesIndexManager, get_edges_values, NodeTypes
from dataset_builder import ReactionDataset
from tagging import ReactionTag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nodes = ["reaction", "entities", "locations", "catalysis", "catalyst_activities", "modifications"]
REACTION = NodeTypes().reaction


class Scorer:
    def __init__(self, name, class_names):
        self.name = name
        self.class_names = class_names
        self.index_to_class = {i: class_name for i, class_name in enumerate(class_names)}
        self.reset()

    def reset(self):
        self.count = 0
        self.loss = 0
        self.index_to_class = {i: class_name for i, class_name in enumerate(self.class_names)}
        for class_name in self.class_names:
            setattr(self, f'{class_name}_acc', 0)
            setattr(self, f'{class_name}_tp', 0)
            setattr(self, f'{class_name}_fp', 0)
            setattr(self, f'{class_name}_fn', 0)
            setattr(self, f'{class_name}_tn', 0)

    def add(self, y, pred, loss):
        self.count += 1
        self.loss += loss
        if y.ndim == 2:
            y = y[0]
        if pred.ndim == 2:
            pred = pred[0]
        for i, class_name in self.index_to_class.items():
            pred_i = pred[i]
            y_i = y[i]
            if pred_i == y_i:
                setattr(self, f'{class_name}_acc', getattr(self, f'{class_name}_acc') + 1)
            if pred_i == 1 and y_i == 1:
                setattr(self, f'{class_name}_tp', getattr(self, f'{class_name}_tp') + 1)
            elif pred_i == 1 and y_i == 0:
                setattr(self, f'{class_name}_fp', getattr(self, f'{class_name}_fp') + 1)
            elif pred_i == 0 and y_i == 1:
                setattr(self, f'{class_name}_fn', getattr(self, f'{class_name}_fn') + 1)
            elif pred_i == 0 and y_i == 0:
                setattr(self, f'{class_name}_tn', getattr(self, f'{class_name}_tn') + 1)
            else:
                raise Exception("Invalid state", pred_i, y_i)

    def __repr__(self):
        res = f"{self.name}\nloss: {self.loss / self.count}\n{self.count} samples\n"
        for class_name in self.class_names:
            acc = getattr(self, f'{class_name}_acc')
            tp = getattr(self, f'{class_name}_tp')
            fp = getattr(self, f'{class_name}_fp')
            fn = getattr(self, f'{class_name}_fn')
            tn = getattr(self, f'{class_name}_tn')
            res += f"{class_name} acc: {acc / self.count}, tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}\n"
        return res


class PartialFixedEmbedding(nn.Module):
    def __init__(self, node_index_manager: NodesIndexManager, learned_embedding_dim):
        super(PartialFixedEmbedding, self).__init__()
        self.node_index_manager=node_index_manager
        self.embeddings = dict()
        for index in range(node_index_manager.index_count):
            node = node_index_manager.index_to_node[index]
            if node.vec is not None:
                self.embeddings[index] = nn.Parameter(torch.tensor(node.vec, dtype=torch.float32).to(device))
            else:
                self.embeddings[index] = nn.Parameter(
                    nn.init.uniform_(torch.zeros(learned_embedding_dim), -1.0, 1.0).to(device))

    def forward(self, input):

        input = input.flatten()
        input = torch.stack([self.embeddings[i.item()] for i in input])
        return input


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, emb_dim=1024, num_layers=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {e: SAGEConv(-1, hidden_channels).to(device) for e in get_edges_values()}, aggr='sum')
            self.convs.append(conv)

        self.lin_reaction = Linear(hidden_channels, out_channels).to(device)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            # print(x_dict.keys())
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin_reaction(x_dict[REACTION])


def run_model(data, model, emb_model, optimizer, scorer: Scorer, is_train=True):
    x_dict = {key: emb_model(data.x_dict[key]) for key in data.x_dict.keys() if key not in ['y', 'output_vec']}
    y = data['tags'].to(device)
    y = y.float()
    y = y.unsqueeze(0)
    edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
    out = model(x_dict, edge_index_dict)
    pred = (out > 0).to(torch.int32)
    loss = nn.BCEWithLogitsLoss()(out, y)
    scorer.add(y.cpu().numpy(), pred.detach().cpu().numpy(), loss.item())
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    root = "data/items"
    dataset = ReactionDataset(sample=100, root=root).reactions

    import random

    random.seed(10)

    # split into train and test:
    random.shuffle(dataset)
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8):]

    node_index_manager = NodesIndexManager(root=root)

    emb_model = PartialFixedEmbedding(node_index_manager, 512).to(device)
    model = HeteroGNN(hidden_channels=512, out_channels=ReactionTag().get_num_tags()).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    classes_names = dataclasses.asdict(ReactionTag()).keys()
    for i in range(100):
        print(f"Epoch {i}")

        train_score = Scorer("train", classes_names)
        for data in tqdm(train_dataset):
            run_model(data, model, emb_model, optimizer, train_score)
        print(train_score)
        test_score = Scorer("test", classes_names)
        for data in tqdm(test_dataset):
            run_model(data, model, emb_model, optimizer, test_score, False)
        print(test_score)
        torch.save(model.state_dict(), f"data/model/model.pt")
        torch.save(emb_model.state_dict(), f"data/model/emb_model.pt")

    print("Training completed!")
