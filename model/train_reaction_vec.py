import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data_types import REACTION
from common.path_manager import model_path, scores_path, item_path
from common.utils import get_last_epoch_model, TYPE_TO_VEC_DIM, TEXT
from dataset.dataset_builder import get_data
from dataset.index_manger import NodesIndexManager
from model.gnn_models import GnnModelConfig, HeteroGNN
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_dim = TYPE_TO_VEC_DIM[TEXT]


class EmdDataset:
    def __init__(self, dataset, model, neg_count=3, return_index=False):
        self.return_index = return_index
        self.all_emd = []
        self.all_vec = []
        self.all_index = []
        self.labels = []

        self.vectors = np.load(f'{item_path}/bp_vec.npy')
        for data in tqdm(dataset):
            data = data.to(device)
            if data.bp.item() == -1:
                continue
            with torch.no_grad():
                x_dict = {key: data.x_dict[key] for key in data.x_dict.keys()}
                edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
                _, emb = model(x_dict, edge_index_dict, return_reaction_embedding=True)
            self.all_emd.append(emb[0].detach().cpu().numpy().flatten())
            self.all_vec.append(self.vectors[data.bp.item()].flatten())
            self.all_index.append(data.bp.item())
            self.labels.append(1)
            for i in range(neg_count):
                idx = np.random.randint(0, len(self.vectors) - 1)
                if idx == data.bp.item():
                    continue
                self.all_emd.append(emb[0].detach().cpu().numpy().flatten())
                self.all_vec.append(self.vectors[idx].flatten())
                self.all_index.append(idx)
                self.labels.append(-1)

    def __len__(self):
        return len(self.all_emd)

    def __getitem__(self, idx):
        if self.return_index:
            return self.all_emd[idx], self.all_vec[idx], self.labels[idx], self.all_index[idx]
        return self.all_emd[idx], self.all_vec[idx], self.labels[idx]


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
    test_emd = EmdDataset(test_dataset, model)
    train_loader = DataLoader(train_emd, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_emd, batch_size=32, shuffle=True)
    return train_loader, test_loader, node_index_manager


def evaluate(model, dataset: EmdDataset):
    model.eval()
    dataset.return_index = True
    indexes = []
    for i in range(len(dataset)):
        emd, _, label, index = dataset[i]

        if label == -1:
            continue
        emd = torch.tensor(emd).to(device)

        with torch.no_grad():
            out = model(emd.unsqueeze(0), REACTION)[0]
        torch_tensor = torch.tensor(dataset.vectors).to(device)
        sim = torch.nn.functional.cosine_similarity(out, torch_tensor)
        sim = sim.cpu().numpy()
        order = list(np.argsort(sim))[::-1]
        indexes.append(order.index(index))
    top_1= np.mean([1 for i in indexes if i < 1])
    top_5 = np.mean([1 for i in indexes if i < 5])
    top_10 = np.mean([1 for i in indexes if i < 10])
    top_50 = np.mean([1 for i in indexes if i < 50])
    print(f"Top 1: {top_1}, Top 5: {top_5}, Top 10: {top_10}, Top 50: {top_50}, Mean Index: {np.mean(indexes)}")

    dataset.return_index = False


def run_epoch(model, data_loader, optimizer, criterion, is_train, epoch, scores_file):
    model.train() if is_train else model.eval()
    total_loss = 0
    for emd, vec, labels in tqdm(data_loader):
        emd, vec, labels = emd.to(device), vec.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(emd, REACTION)
            loss = criterion(out, vec, labels)
            total_loss += loss.item()
            if is_train:
                loss.backward()
                optimizer.step()
    loss_mean = total_loss / len(data_loader)
    msg = f"Epoch {epoch} {'Train' if is_train else 'Test'} Loss: {loss_mean:.4f}"
    print(msg)
    with open(scores_file, "a") as f:
        f.write(msg + "\n")
    evaluate(model, data_loader.dataset)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gnn_default")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    model, config = load_model(args.model_name)
    model = model.to(device)
    train_loader, test_loader, node_index_manager = load_data(model)

    save_dir = f"{model_path}/reaction-vec_{args.model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        for f in os.listdir(save_dir):
            if f.endswith(".pt"):
                os.remove(f"{save_dir}/{f}")
    scores_file = f"{scores_path}/reaction-vec_{args.model_name}.txt"
    if os.path.exists(scores_file):
        os.remove(scores_file)
    n_bp = len(node_index_manager.bp_name_to_index)
    classify_config = MultiModalLinearConfig(
        embedding_dim=[config.hidden_channels], n_layers=args.n_layers,
        names=[REACTION], hidden_dim=args.hidden_dim,
        output_dim=[text_dim], dropout=args.dropout, normalize_last=1
    )
    classify_config.save_to_file(f"{save_dir}/config.txt")

    classify_model = MiltyModalLinear(classify_config).to(device)
    classify_model = classify_model.to(device)
    optimizer = torch.optim.Adam(classify_model.parameters(), lr=0.001)
    criterion = torch.nn.CosineEmbeddingLoss()

    for epoch in range(args.epochs):
        run_epoch(classify_model, train_loader, optimizer, criterion, True, epoch, scores_file)
        run_epoch(classify_model, test_loader, optimizer, criterion, False, epoch, scores_file)
        torch.save(classify_model.state_dict(), f"{save_dir}/epoch_{epoch}.pt")
