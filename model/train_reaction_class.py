import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data_types import REACTION
from common.path_manager import model_path, scores_path
from common.utils import get_last_epoch_model
from dataset.dataset_builder import get_data
from dataset.index_manger import NodesIndexManager
from model.gnn_models import GnnModelConfig, HeteroGNN
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmdDataset:
    def __init__(self, dataset, model, filter_tags=None):
        self.all_emd = []
        self.labels = []
        for data in tqdm(dataset):
            data = data.to(device)
            if data.bp.item() == -1 or (filter_tags is not None and data.bp.item() not in filter_tags):
                continue
            with torch.no_grad():
                x_dict = {key: data.x_dict[key] for key in data.x_dict.keys()}
                edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
                _, emb = model(x_dict, edge_index_dict, return_reaction_embedding=True)
            self.all_emd.append(emb[0].detach().cpu().numpy().flatten())
            self.labels.append(data.bp.item())

    def __len__(self):
        return len(self.all_emd)

    def __getitem__(self, idx):
        return self.all_emd[idx], self.labels[idx]


def load_model(model_name):
    model_dir = f"{model_path}/{model_name}"
    cp_name = get_last_epoch_model(model_dir)
    config = GnnModelConfig.load_from_file(f"{model_dir}/config.txt")
    model = HeteroGNN(config)
    model.load_state_dict(torch.load(cp_name, map_location=torch.device('cpu')))
    model.eval()
    return model, config


def load_data(model):
    node_index_manager = NodesIndexManager()
    train_dataset, test_dataset, _ = get_data(node_index_manager, fake_task=True, location_augmentation_factor=0,
                                              entity_augmentation_factor=0)

    train_emd = EmdDataset(train_dataset, model)
    train_tags = np.unique([d[1] for d in train_emd])
    test_emd = EmdDataset(test_dataset, model, train_tags)
    n_bp = len(node_index_manager.bp_name_to_index)
    labels = torch.LongTensor(list(range(n_bp)) + [d[1] for d in train_emd])

    counts = torch.bincount(labels)
    weights = 1 / counts

    train_loader = DataLoader(train_emd, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_emd, batch_size=32, shuffle=True)
    return train_loader, test_loader, node_index_manager, weights


def run_epoch(model, data_loader, optimizer, criterion, is_train, epoch, n_bp, scores_file, top_k=3):
    model.train() if is_train else model.eval()
    per_label_acc = defaultdict(list)
    all_loss = []
    for emd, labels in tqdm(data_loader):
        emd, labels = emd.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(emd, REACTION)

            loss = criterion(out, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        all_loss.append(loss.item())

        # Calculate top-k accuracy
        _, topk_preds = out.topk(top_k, dim=-1)
        correct_topk = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))

        acc = correct_topk.sum().item() / len(labels)
        per_label_acc["all"].append(acc)

        for i in range(n_bp):
            mask = labels == i
            if mask.sum() > 0:
                acc = correct_topk[mask].sum().item() / mask.sum().item()
                per_label_acc[i].append(acc)

    per_label_acc = {k: np.mean(v) for k, v in per_label_acc.items()}
    per_label_mean = np.mean([v for k, v in per_label_acc.items() if k != "all"])
    all_mean = per_label_acc["all"]
    skip_count = n_bp - len(per_label_acc)
    loss_mean = np.mean(all_loss)
    train_or_test = "train" if is_train else "test"

    msg = f"{epoch} {train_or_test} loss: {loss_mean} all: {all_mean} mean: {per_label_mean} skipped: {skip_count}"
    print(msg)
    with open(scores_file, "a") as f:
        f.write(msg + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gnn_default")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    model, config = load_model(args.model_name)
    model = model.to(device)
    train_loader, test_loader, node_index_manager, weights = load_data(model)

    save_dir = f"{model_path}/reaction_{args.model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for f in os.listdir(save_dir):
            if f.endswith(".pt"):
                os.remove(f"{save_dir}/{f}")
    scores_file = f"{scores_path}/reaction_{args.model_name}.txt"
    if os.path.exists(scores_file):
        os.remove(scores_file)
    n_bp = len(node_index_manager.bp_name_to_index)
    classify_config = MultiModalLinearConfig(
        embedding_dim=[config.hidden_channels], n_layers=args.n_layers,
        names=[REACTION], hidden_dim=args.hidden_dim,
        output_dim=[n_bp], dropout=args.dropout, normalize_last=0
    )
    classify_config.save_to_file(f"{save_dir}/config.txt")

    classify_model = MiltyModalLinear(classify_config).to(device)
    classify_model = classify_model.to(device)
    optimizer = torch.optim.Adam(classify_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    for epoch in range(args.epochs):
        run_epoch(classify_model, train_loader, optimizer, criterion, True, epoch, n_bp, scores_file)
        run_epoch(classify_model, test_loader, optimizer, criterion, False, epoch, n_bp, scores_file)
        torch.save(classify_model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")
