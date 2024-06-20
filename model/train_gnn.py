import dataclasses
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from common.scorer import Scorer
from dataset.index_manger import NodesIndexManager
from common.data_types import NodeTypes, REAL, FAKE_LOCATION_ALL, FAKE_PROTEIN, FAKE_MOLECULE
from dataset.dataset_builder import get_data,data_to_batches
from model.gnn_models import GnnModelConfig, HeteroGNN
from tagging import ReactionTag
from torch_geometric.loader import DataLoader
import seaborn as sns
from common.path_manager import scores_path, model_path
import random

sns.set()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REACTION = NodeTypes().reaction

batch_size = 2048


def run_model(data, model, optimizer, scorer, is_train=True):
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
    prev_time = time.time()
    for i in range(epochs):

        train_score = Scorer("train", scores_tag_names)
        # train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_data = data_to_batches(train_dataset, batch_size,True)
        for data_index, data in enumerate(train_data):
            run_model(data, model, optimizer, train_score)
        log_func(train_score.get_log(), i)
        test_score = Scorer("test", scores_tag_names)
        # test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_data = data_to_batches(test_dataset, batch_size, False)

        for data_index, data in enumerate(test_data):
            run_model(data, model, optimizer, test_score, False)
        log_func(test_score.get_log(), i)
        print("Finished epoch", i, "time:", (time.time() - prev_time) / 60, "minutes")
        prev_time = time.time()
        name = f'{save_dir}/model_{i}.pt'
        torch.save(model.state_dict(), name)
        # torch.save(optimizer.state_dict(), name.replace("model_", "optimizer_"))


def args_to_config(args):
    return GnnModelConfig(
        learned_embedding_dim=args.gnn_learned_embedding_dim,
        hidden_channels=args.gnn_hidden_channels,
        num_layers=args.gnn_num_layers,
        conv_type=args.gnn_conv_type,
        train_all_emd=args.gnn_train_all_emd,
        fake_task=args.gnn_fake_task,
        pretrained_method=args.gnn_pretrained_method,
        fuse_name=args.fuse_name,
        out_channels=args.gnn_out_channels,
        last_or_concat=args.gnn_last_or_concat,
        # reaction_or_mean=args.gnn_reaction_or_mean
    )


def run_with_args(args):
    save_dir = f"{model_path}/gnn_{args.name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(save_dir):
        if file.endswith(".pt"):
            os.remove(f"{save_dir}/{file}")

    score_file = f"{scores_path}/gnn_{args.name}.txt"
    if os.path.exists(score_file):
        os.remove(score_file)

    def save_to_file(x, step):
        with open(score_file, "a") as f:
            f.write(f"{step}\n")
            f.write(f"{x}\n")

    config = args_to_config(args)
    config.save_to_file(f"{save_dir}/config.txt")
    model = HeteroGNN(args_to_config(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)

    train(model, optimizer, batch_size, save_to_file, args.gnn_epochs, save_dir=save_dir)


if __name__ == "__main__":
    from common.args_manager import get_args

    args = get_args()

    args.gnn_out_channels = 1 if args.gnn_fake_task else 6
    if args.gnn_fake_task:
        tag_names = ["fake"]
        scores_tag_names = [REAL, FAKE_LOCATION_ALL, FAKE_PROTEIN, FAKE_MOLECULE]
    else:
        tag_names = [x for x in dataclasses.asdict(ReactionTag()).keys() if x != "fake"]
        scores_tag_names = tag_names
    node_index_manager = NodesIndexManager(pretrained_method=args.gnn_pretrained_method, fuse_name=args.fuse_name)
    train_dataset, test_dataset, _, pos_classes_weights = get_data(node_index_manager, sample=args.gnn_sample,
                                                                   fake_task=args.gnn_fake_task, data_aug=args.data_aug)
    pos_classes_weights = pos_classes_weights.to(device)
    run_with_args(args)
