import dataclasses
import time

import pandas as pd
import torch
import torch.nn as nn
import os
from common.scorer import Scorer
from dataset.index_manger import NodesIndexManager
from common.data_types import NodeTypes, REAL, FAKE_LOCATION_ALL, FAKE_PROTEIN, FAKE_MOLECULE
from common.utils import prepare_files
from dataset.dataset_builder import get_data, data_to_batches
from model.gnn_models import GnnModelConfig, HeteroGNN
from tagging import ReactionTag
import seaborn as sns
from common.path_manager import scores_path, model_path

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
        train_data = data_to_batches(train_dataset, batch_size, True)
        for data_index, data in enumerate(train_data):
            run_model(data, model, optimizer, train_score)
        log_func(train_score.get_log(), i)

        valid_score = Scorer("valid", scores_tag_names)
        # valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        valid_data = data_to_batches(valid_dataset, batch_size, False)
        for data_index, data in enumerate(valid_data):
            run_model(data, model, optimizer, valid_score, False)
        log_func(valid_score.get_log(), i)

        test_score = Scorer("test", scores_tag_names)
        # test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_data = data_to_batches(test_dataset, batch_size, False)
        for data_index, data in enumerate(test_data):
            run_model(data, model, optimizer, test_score, False)
        log_func(test_score.get_log(), i)

        print("Finished epoch", i, "time:", time.time() - prev_time, "seconds")
        prev_time = time.time()
        # name = f'{save_dir}/model_{i}.pt'
        # torch.save(model.state_dict(), name)
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


def print_best_results(results_file):
    with open(results_file, "r") as f:
        lines = f.readlines()
    valid_results = pd.DataFrame(columns=['all', 'protein', 'molecule', 'location'])
    test_results = pd.DataFrame(columns=['all', 'protein', 'molecule', 'location'])
    for i in range(0, len(lines), 6):  # num,train,num,valid,num,test
        valid_scores = eval(lines[i + 3].replace("nan", "0"))
        valid_scores = {key.split("_")[-1]: valid_scores[key] for key in valid_scores if "_" in key}
        test_scores = eval(lines[i + 5].replace("nan", "0"))
        test_scores = {key.split("_")[-1]: test_scores[key] for key in test_scores if "_" in key}
        valid_results.loc[i // 6] = valid_scores
        test_results.loc[i // 6] = test_scores
    print("Valid results")
    print(valid_results)
    print("Test results")
    print(test_results)
    # choose the best index for each column based on the valid results
    best_index = valid_results.idxmax()
    for col in valid_results.columns:
        print(f"Best model for {col}")
        print(test_results.loc[best_index[col]])
    name = os.path.basename(results_file).replace(".txt", "")
    summary = [name] + list(test_results.loc[best_index['protein']].values)
    summary = ",".join([str(x) for x in summary])
    if not os.path.exists(f"{scores_path}/summary_gnn.csv"):
        with open(f"{scores_path}/summary_gnn.csv", "w") as f:
            f.write(",".join(["name"] + list(test_results.columns)) + "\n")
    with open(f"{scores_path}/summary_gnn.csv", "a") as f:
        f.write(summary + "\n")


def run_with_args(args):
    save_dir, score_file = prepare_files(f'gnn_{args.name}')

    def save_to_file(x, step):
        with open(score_file, "a") as f:
            f.write(f"{step}\n")
            f.write(f"{x}\n")

    config = args_to_config(args)
    config.save_to_file(f"{save_dir}/config.txt")
    model = HeteroGNN(args_to_config(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)

    train(model, optimizer, batch_size, save_to_file, args.gnn_epochs, save_dir=save_dir)
    print_best_results(score_file)


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

    node_index_manager = NodesIndexManager(pretrained_method=args.gnn_pretrained_method, fuse_name=args.fuse_name,
                                           fuse_pretrained_start=args.fuse_pretrained_start)

    train_dataset, valid_dataset, test_dataset, pos_classes_weights = get_data(node_index_manager,
                                                                               sample=args.gnn_sample,
                                                                               fake_task=args.gnn_fake_task,
                                                                               data_aug=args.data_aug)
    pos_classes_weights = pos_classes_weights.to(device)
    run_with_args(args)
