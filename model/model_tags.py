import dataclasses

from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv, SAGEConv, Linear
import numpy as np

from common.scorer import Scorer
from dataset.index_manger import NodesIndexManager, PRETRAINED_EMD
from common.utils import get_edges_values
from common.data_types import NodeTypes, DATA_TYPES, EMBEDDING_DATA_TYPES, REAL, FAKE_LOCATION_ALL, \
    FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE
from dataset.dataset_builder import get_data
from tagging import ReactionTag
from collections import defaultdict
from torch_geometric.loader import DataLoader
import seaborn as sns
from common.path_manager import scores_path, model_path

sns.set()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REACTION = NodeTypes().reaction

batch_size = 1

conv_type_to_class = {
    "TransformerConv": TransformerConv,
    "SAGEConv": SAGEConv
}

conv_type_to_args = {
    "TransformerConv": {},
    "SAGEConv": {"normalize": True}
}


@dataclasses.dataclass
class GnnModelConfig:
    learned_embedding_dim: int
    hidden_channels: int
    num_layers: int
    conv_type: str
    train_all_emd: int
    fake_task: int
    pretrained_method: int
    fuse_config: str
    out_channels: int
    return_reaction_embedding: int=False


class PartialFixedEmbedding(nn.Module):
    def __init__(self, node_index_manager: NodesIndexManager, learned_embedding_dim, train_all_emd):
        super(PartialFixedEmbedding, self).__init__()
        self.node_index_manager = node_index_manager
        self.embeddings = nn.ModuleDict()
        self.embeddings[NodeTypes.reaction] = nn.Embedding(1, learned_embedding_dim).to(device)
        self.embeddings[NodeTypes.complex] = nn.Embedding(1, learned_embedding_dim).to(device)
        for dtype in DATA_TYPES:
            if dtype in EMBEDDING_DATA_TYPES:
                dtype_indexes = range(node_index_manager.dtype_to_first_index[dtype],
                                      node_index_manager.dtype_to_last_index[dtype])
                vectors = np.stack([node_index_manager.index_to_node[i].vec for i in dtype_indexes])
                self.embeddings[dtype] = nn.Embedding.from_pretrained(
                    torch.tensor(vectors, dtype=torch.float32).to(device))
                if not train_all_emd:
                    self.embeddings[dtype].requires_grad_(False)
                else:
                    self.embeddings[dtype].requires_grad_(True)
            else:
                n_samples = node_index_manager.dtype_to_last_index[dtype] - node_index_manager.dtype_to_first_index[
                    dtype]
                self.embeddings[dtype] = nn.Embedding(n_samples, learned_embedding_dim).to(device)

    def forward(self, x):
        x = x.squeeze(1)
        dtype = self.node_index_manager.index_to_node[x[0].item()].type
        if dtype == NodeTypes.reaction or dtype == NodeTypes.complex:
            return self.embeddings[dtype](torch.zeros_like(x).to(device))
        x = x - self.node_index_manager.dtype_to_first_index[dtype]
        x = self.embeddings[dtype](x)
        return x


class HeteroGNN(torch.nn.Module):
    def __init__(self, config: GnnModelConfig):

        super().__init__()
        node_index_manager = NodesIndexManager(fuse_vec=config.pretrained_method, fuse_config=config.fuse_config)

        self.emb = PartialFixedEmbedding(node_index_manager, config.learned_embedding_dim, config.train_all_emd).to(
            device)
        self.convs = torch.nn.ModuleList()
        self.save_activations = defaultdict(list)

        for i in range(config.num_layers):
            conv_per_edge = {}
            for edge in get_edges_values():
                e_conv = conv_type_to_class[config.conv_type](-1, config.hidden_channels,
                                                              **conv_type_to_args[config.conv_type]).to(
                    device)
                conv_per_edge[edge] = e_conv
            conv = HeteroConv(conv_per_edge, aggr='sum')
            self.convs.append(conv)
        self.lin_reaction = Linear(config.hidden_channels, config.out_channels).to(device)
        self.return_reaction_embedding = config.return_reaction_embedding

    def forward(self, x_dict, edge_index_dict):
        for key, x in x_dict.items():
            x_dict[key] = self.emb(x)
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.return_reaction_embedding:
            return self.lin_reaction(x_dict[REACTION]), x_dict[REACTION]
        return self.lin_reaction(x_dict[REACTION])


def run_model(data, model, optimizer, scorer: Scorer, is_train=True):
    x_dict = {key: data.x_dict[key].to(device) for key in data.x_dict.keys()}
    y = data['tags'].to(device)
    augmentation_types = data['augmentation_type']
    y = y.float()
    edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
    out = model(x_dict, edge_index_dict)
    pred = (out > 0).to(torch.int32)
    y = y.reshape(out.shape)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_classes_weights)(out, y)

    fake_task = len(y[0]) == 1

    scorer.add(y.cpu().numpy(), pred.detach().cpu().numpy(), out.detach().cpu().numpy(), loss.item(),
               class_names=augmentation_types if fake_task else None)
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(model, optimizer, batch_size, log_func, epochs, save_model=""):
    for i in range(epochs):

        train_score = Scorer("train", scores_tag_names)
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for data_index, data in tqdm(enumerate(train_data)):
            run_model(data, model, optimizer, train_score)
        log_func(train_score.get_log(), i)
        test_score = Scorer("test", scores_tag_names)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for data_index, data in tqdm(enumerate(test_data)):
            run_model(data, model, optimizer, test_score, False)
        log_func(test_score.get_log(), i)
        if save_model:
            name = f'{model_path}/model_{save_model}_{i}.pt'
            torch.save(model.state_dict(), name)
            torch.save(optimizer.state_dict(), name.replace("model_", "optimizer_"))


def args_to_str(args):
    args_dict = vars(args)
    sorted_args = sorted(args_dict.items())
    args_str = '-'.join([f"{key}_{value}" for key, value in sorted_args])
    return args_str


def args_to_config(args):
    return GnnModelConfig(
        learned_embedding_dim=args.learned_embedding_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        train_all_emd=args.train_all_emd,
        fake_task=args.fake_task,
        pretrained_method=args.pretrained_method,
        fuse_config=args.fuse_config,
        out_channels=args.out_channels,
        return_reaction_embedding=args.return_reaction_embedding,
    )


def run_with_args(args):
    run_name = args_to_str(args)
    file_name = f"{scores_path}/{run_name}.txt"

    def save_to_file(x, step):
        with open(file_name, "a") as f:
            f.write(f"{step}\n")
            f.write(f"{x}\n")

    model = HeteroGNN(args_to_config(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, batch_size, save_to_file, args.epochs, save_model=run_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learned_embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--conv_type", type=str, default="SAGEConv", choices=["SAGEConv", "TransformerConv"])
    parser.add_argument("--pretrained_method", type=int, default=PRETRAINED_EMD)
    parser.add_argument("--train_all_emd", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--fake_task", type=int, default=1)
    parser.add_argument("--fuse_config", type=str, default="8192_1_1024_0.0_0.001_1_512")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--return_reaction_embedding", type=int, default=0)


    args = parser.parse_args()
    args.out_channels = 1 if args.fake_task else 6
    if args.fake_task:
        tag_names = ["fake"]
        scores_tag_names = [REAL, FAKE_LOCATION_ALL, FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE]
    else:
        tag_names = [x for x in dataclasses.asdict(ReactionTag()).keys() if x != "fake"]
        scores_tag_names = tag_names
    node_index_manager = NodesIndexManager(fuse_vec=args.pretrained_method, fuse_config=args.fuse_config)
    train_dataset, test_dataset, pos_classes_weights = get_data(node_index_manager, sample=args.sample,
                                                                fake_task=args.fake_task)
    pos_classes_weights = pos_classes_weights.to(device)
    run_with_args(args)
