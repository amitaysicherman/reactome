import dataclasses

from tqdm import tqdm
import torch
import torch.nn as nn
import os
from common.scorer import Scorer
from dataset.index_manger import NodesIndexManager, PRETRAINED_EMD
from common.data_types import NodeTypes, REAL, FAKE_LOCATION_ALL, \
    FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE
from dataset.dataset_builder import get_data
from model.gnn_models import GnnModelConfig, HeteroGNN
from tagging import ReactionTag
from torch_geometric.loader import DataLoader
import seaborn as sns
from common.path_manager import scores_path, model_path

sns.set()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REACTION = NodeTypes().reaction

batch_size = 1


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


def train(model, optimizer, batch_size, log_func, epochs, save_dir=""):
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

        name = f'{save_dir}/model_{i}.pt'
        torch.save(model.state_dict(), name)
        torch.save(optimizer.state_dict(), name.replace("model_", "optimizer_"))


def args_to_config(args):
    return GnnModelConfig(
        learned_embedding_dim=args.learned_embedding_dim,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        train_all_emd=args.train_all_emd,
        fake_task=args.fake_task,
        pretrained_method=args.pretrained_method,
        fuse_name=args.fuse_name,
        out_channels=args.out_channels,
    )


def run_with_args(args):
    save_dir = f"{model_path}/gnn_{args.name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(save_dir):
        os.remove(f"{save_dir}/{file}")

    score_file = f"{scores_path}/gnn_{args.name}.txt"

    def save_to_file(x, step):
        with open(score_file, "a") as f:
            f.write(f"{step}\n")
            f.write(f"{x}\n")

    model = HeteroGNN(args_to_config(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, batch_size, save_to_file, args.epochs, save_dir=save_dir)


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
    parser.add_argument("--fuse_name", type=str, default="all-recon")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--name", type=str, default="default")

    args = parser.parse_args()
    args.out_channels = 1 if args.fake_task else 6
    if args.fake_task:
        tag_names = ["fake"]
        scores_tag_names = [REAL, FAKE_LOCATION_ALL, FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE]
    else:
        tag_names = [x for x in dataclasses.asdict(ReactionTag()).keys() if x != "fake"]
        scores_tag_names = tag_names
    node_index_manager = NodesIndexManager(pretrained_method=args.pretrained_method, fuse_name=args.fuse_name)
    train_dataset, test_dataset, pos_classes_weights = get_data(node_index_manager, sample=args.sample,
                                                                fake_task=args.fake_task)
    pos_classes_weights = pos_classes_weights.to(device)
    run_with_args(args)
