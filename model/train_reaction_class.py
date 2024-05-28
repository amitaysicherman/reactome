import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from dataset.dataset_builder import get_data
from torch.utils.data import DataLoader
from common.utils import get_last_epoch_model
from dataset.index_manger import NodesIndexManager
from model.gnn_models import GnnModelConfig, HeteroGNN
from model.models import MultiModalLinearConfig, MiltyModalLinear
from common.path_manager import model_path, scores_path
from common.data_types import REACTION
from tqdm import tqdm
import os

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
    test_loader = DataLoader(test_emd, batch_size=1, shuffle=True)
    return train_loader, test_loader, node_index_manager, weights


def run_epoch(model, data_loader, optimizer, criterion, is_train, epoch, n_bp, scores_file):
    preds = []
    real = []

    model.train() if is_train else model.eval()

    for emd, labels in tqdm(data_loader):
        emd, labels = emd.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(emd, REACTION)
            loss = criterion(out, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        probs = torch.nn.functional.softmax(out, dim=-1)
        preds.extend(probs.detach().cpu().numpy().tolist())
        real.extend([[0] * l + [1] + [0] * (n_bp - l - 1) for l in labels.detach().cpu().numpy().tolist()])

    real, preds = np.array(real), np.array(preds)
    auc_scores = [roc_auc_score(real[:, i], preds[:, i]) for i in range(n_bp) if len(np.unique(real[:, i])) > 1]

    skip_count = n_bp - len(auc_scores)
    train_or_test = "train" if is_train else "test"
    print(f"Epoch {epoch} ({train_or_test}) AUC: {np.mean(auc_scores):.4f} skip: {skip_count}/{n_bp}")
    with open(scores_file, "a") as f:
        f.write(f"Epoch {epoch} ({train_or_test}) AUC: {np.mean(auc_scores):.4f} skip: {skip_count}/{n_bp}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gnn_default")
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
        embedding_dim=[config.hidden_channels], n_layers=1,
        names=[REACTION], hidden_dim=config.hidden_channels,
        output_dim=[n_bp], dropout=0.0, normalize_last=0
    )
    classify_config.save_to_file(f"{save_dir}/config.txt")

    classify_model = MiltyModalLinear(classify_config).to(device)
    classify_model = classify_model.to(device)
    optimizer = torch.optim.Adam(classify_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    for epoch in range(100):
        run_epoch(classify_model, train_loader, optimizer, criterion, True, epoch, n_bp, scores_file)
        run_epoch(classify_model, test_loader, optimizer, criterion, False, epoch, n_bp, scores_file)
        torch.save(classify_model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")
