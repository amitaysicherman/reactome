import dataclasses
import time

import pandas as pd
import torch
import torch.nn as nn
import os
from common.scorer import Scorer
from dataset.index_manger import NodesIndexManager, get_from_args
from common.data_types import NodeTypes, REAL, FAKE_LOCATION_ALL, FAKE_PROTEIN, FAKE_MOLECULE, FAKE_TEXT
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
    if is_train:
        model.train()
    else:
        model.eval()
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
               class_names=augmentation_types if fake_task else None, id_list=data['id_'].cpu().numpy().tolist())
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run_epoch(dataset, model, optimizer, name, scores_tag_names, batch_size, log_func, i, full_output_path=""):
    scorer = Scorer(name, scores_tag_names)
    # batch_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_data = data_to_batches(dataset, batch_size, True if name == "train" else False)
    for data_index, data in enumerate(batch_data):
        run_model(data, model, optimizer, scorer, is_train=True if name == "train" else False)
    if full_output_path:
        scorer.save_full_res(full_output_path)
    log_func(scorer.get_log(), i)


def train(model, optimizer, batch_size, log_func, epochs, save_dir="", score_file=""):
    prev_time = time.time()
    for i in range(epochs):
        output_file_prefix = f"{score_file.replace('.txt', '')}_{i}"
        run_epoch(train_dataset, model, optimizer, "train", scores_tag_names, batch_size, log_func, i,
                  output_file_prefix)
        run_epoch(valid_dataset, model, optimizer, "valid", scores_tag_names, batch_size, log_func, i,
                  output_file_prefix)
        run_epoch(test_dataset, model, optimizer, "test", scores_tag_names, batch_size, log_func, i, output_file_prefix)

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
        # reaction_or_mean=args.gnn_reaction_or_mean,
        fuse_pretrained_start=args.fuse_pretrained_start,
        drop_out=args.gnn_drop_out
    )


def print_best_results(results_file):
    with open(results_file, "r") as f:
        lines = f.readlines()
    valid_results = pd.DataFrame(columns=['all', 'protein', 'molecule', 'location', 'text'])
    test_results = pd.DataFrame(columns=['all', 'protein', 'molecule', 'location', 'text'])
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

    train(model, optimizer, batch_size, save_to_file, args.gnn_epochs, save_dir=save_dir, score_file=score_file)
    print_best_results(score_file)


if __name__ == "__main__":
    from common.args_manager import get_args

    args = get_args()

    args.gnn_out_channels = 1 if args.gnn_fake_task else 6
    if args.gnn_fake_task:
        tag_names = ["fake"]
        scores_tag_names = [REAL, FAKE_LOCATION_ALL, FAKE_PROTEIN, FAKE_MOLECULE, FAKE_TEXT]
    else:
        tag_names = [x for x in dataclasses.asdict(ReactionTag()).keys() if x != "fake"]
        scores_tag_names = tag_names

    node_index_manager: NodesIndexManager = get_from_args(args)
    filter_untrain = False
    if args.gnn_pretrained_method == 0:
        filter_untrain = True
    if args.fuse_pretrained_start == 0:
        filter_untrain = True

    train_dataset, valid_dataset, test_dataset, pos_classes_weights = get_data(node_index_manager,
                                                                               sample=args.gnn_sample,
                                                                               fake_task=args.gnn_fake_task,
                                                                               data_aug=args.data_aug,
                                                                               filter_untrain=filter_untrain)
    pos_classes_weights = pos_classes_weights.to(device)
    run_with_args(args)
