import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.path_manager import data_path
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from model.models import MultiModalLinearConfig, MiltyModalLinear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MOL_DIM = 768
PROT_DIM = 1024


def load_data():
    base_dir = os.path.join(data_path, "protein_drug")
    molecules = np.load(os.path.join(base_dir, "molecules.npy"))[:, 0, :]
    proteins = np.load(os.path.join(base_dir, "proteins.npy"))[:, 0, :]
    with open(os.path.join(base_dir, "molecules.txt")) as f:
        molecules_names = f.read().splitlines()
    with open(os.path.join(base_dir, "proteins.txt")) as f:
        proteins_names = f.read().splitlines()
    with open(os.path.join(base_dir, "labels.txt")) as f:
        labels = f.read().splitlines()
    molecules_names = np.array(molecules_names)
    proteins_names = np.array(proteins_names)
    labels = np.array(labels, dtype=np.float32)
    return molecules, proteins, labels, molecules_names, proteins_names


def split_train_val_test(data, val_size=0.16, test_size=0.20):
    train_val_index = int((1 - val_size - test_size) * len(data))
    val_test_index = int((1 - test_size) * len(data))
    train_data = data[:train_val_index]
    val_data = data[train_val_index:val_test_index]
    test_data = data[val_test_index:]
    return train_data, val_data, test_data


class ProteinDrugDataset(Dataset):
    def __init__(self, molecules, proteins, labels, e_types=None):
        self.molecules = molecules
        self.proteins = proteins
        self.labels = labels
        self.e_types = e_types if e_types is not None else [0] * len(molecules)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.molecules[idx], self.proteins[idx], self.labels[idx], self.e_types[idx]


def load_fuse_model(base_dir):
    cp_names = os.listdir(base_dir)
    cp_name = [x for x in cp_names if x.endswith(".pt")][0]
    cp_data = torch.load(f"{base_dir}/{cp_name}", map_location=torch.device('cpu'))
    config_file = os.path.join(base_dir, 'config.txt')
    config = MultiModalLinearConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = MiltyModalLinear(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


def data_to_loader(molecules, proteins, labels, e_types, batch_size, shuffle=True):
    dataset = ProteinDrugDataset(molecules, proteins, labels, e_types)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_layers(dims):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
    return layers


class ProteinDrugLinearModel(torch.nn.Module):
    def __init__(self, fuse_base, m_fuse=True, p_fuse=True,
                 m_model=True, p_model=True, only_rand=False, fuse_freeze=True):
        super().__init__()
        self.molecule_dim = 0
        self.protein_dim = 0
        self.m_fuse = m_fuse
        self.p_fuse = p_fuse
        self.m_model = m_model
        self.p_model = p_model

        self.fuse_freeze = fuse_freeze
        if m_fuse or p_fuse:
            self.fuse_model, dim = load_fuse_model(fuse_base)
            self.m_type = "molecule_protein" if "molecule_protein" in self.fuse_model.names else "molecule"
            self.p_type = "protein_protein" if "protein_protein" in self.fuse_model.names else "protein"
            if m_fuse:
                self.molecule_dim += dim
            if p_fuse:
                self.protein_dim += dim

        if m_model:
            self.molecule_dim += MOL_DIM
        if p_model:
            self.protein_dim += PROT_DIM
        self.only_rand = only_rand
        self.m_layers = get_layers([self.molecule_dim, 1024, 512, 256, 128])
        self.p_layers = get_layers([self.protein_dim, 1024, 512, 256, 128])

    def forward(self, molecule, protein):
        if self.m_fuse:
            fuse_molecule = self.fuse_model(molecule, self.m_type)
            if self.fuse_freeze:
                fuse_molecule = fuse_molecule.detach()
            if self.m_model:
                molecule = torch.cat([fuse_molecule, molecule], dim=1)
            else:
                molecule = fuse_molecule
        if self.p_fuse:
            fuse_protein = self.fuse_model(protein, self.p_type)
            if self.fuse_freeze:
                fuse_protein = fuse_protein.detach()
            if self.p_model:
                protein = torch.cat([fuse_protein, protein], dim=1)
            else:
                protein = fuse_protein
        if self.only_rand:
            molecule = torch.randn_like(molecule)
            protein = torch.randn_like(protein)
        molecule = self.m_layers(molecule)
        protein = self.p_layers(protein)
        return -1 * F.cosine_similarity(molecule, protein).unsqueeze(1)


def run_epoch(model, loader, optimizer, criterion, part):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = defaultdict(list)
    preds = defaultdict(list)
    total_loss = 0
    for molecules, proteins, labels, e_type in loader:
        molecules = molecules.to(device)
        proteins = proteins.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(molecules, proteins)
        loss = criterion(outputs, labels.unsqueeze(1))
        if part == "train":
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        for e, r, p in zip(e_type, labels, outputs):
            reals[e.item()].append(r.item())
            preds[e.item()].append(torch.sigmoid(p).item())

    auc = defaultdict(float)
    for key in reals.keys():
        if len(reals[key]) < 100:
            continue
        auc[key] = roc_auc_score(reals[key], preds[key])
    return auc


def get_test_e_type(train_proteins_names, test_proteins_names, train_molecules_names, test_molecules_names):
    train_p_set = set(train_proteins_names)
    train_m_set = set(train_molecules_names)
    test_types = []
    for p, m in zip(test_proteins_names, test_molecules_names):
        if p in train_p_set and m in train_m_set:
            test_types.append(1)
        elif p in train_p_set:
            test_types.append(2)
        elif m in train_m_set:
            test_types.append(3)
        else:
            test_types.append(4)
    return test_types


def score_to_str(score_dict):
    if 0 in score_dict:
        return f'{score_dict[0] * 100:.2f}'
    else:
        return f'{score_dict[1] * 100:.2f}'
    # return " ".join([f"{k}:{v * 100:.1f}" for k, v in score_dict.items()])


def model_to_conf_name(model):
    return f"{model.m_fuse},{model.p_fuse},{model.m_model},{model.p_model},{model.only_rand}"


def get_all_args_opt():
    conf = []
    for m_fuse in [0, 1]:
        for p_fuse in [0, 1]:
            for m_model in [0, 1]:
                for p_model in [0, 1]:
                    for only_rand in [0, 1]:
                        for fuse_freeze in [0, 1]:
                                    conf.append(
                                        f"--m_fuse {m_fuse} --p_fuse {p_fuse} --m_model {m_model} --p_model {p_model} --only_rand {only_rand} --fuse-freeze {fuse_freeze}")
    return conf

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fuse_base", type=str, default="data/models_checkpoints/fuse_all-to-prot")
    parser.add_argument("--fuse_name", type=str, default="fuse_47.pt")
    parser.add_argument("--m_fuse", type=int, default=0)
    parser.add_argument("--p_fuse", type=int, default=0)
    parser.add_argument("--m_model", type=int, default=0)
    parser.add_argument("--p_model", type=int, default=0)
    parser.add_argument("--only_rand", type=int, default=0)
    parser.add_argument("--fuse-freeze", type=int, default=0)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    print(args)
    3/0
    fuse_base = args.fuse_base
    fuse_name = args.fuse_name
    m_fuse = bool(args.m_fuse)
    p_fuse = bool(args.p_fuse)
    m_model = bool(args.m_model)
    p_model = bool(args.p_model)
    only_rand = bool(args.only_rand)
    fuse_freeze = bool(args.fuse_freeze)
    bs = args.bs
    lr = args.lr

    np.random.seed(42)
    all_molecules, all_proteins, all_labels, molecules_names, proteins_names = load_data()
    shuffle_index = np.random.permutation(len(all_molecules))
    all_molecules = all_molecules[shuffle_index]
    all_proteins = all_proteins[shuffle_index]
    all_labels = all_labels[shuffle_index]
    molecules_names = molecules_names[shuffle_index]
    proteins_names = proteins_names[shuffle_index]
    print(all_molecules.shape, all_proteins.shape, all_labels.shape)
    train_molecules, val_molecules, test_molecules = split_train_val_test(all_molecules)
    train_proteins, val_proteins, test_proteins = split_train_val_test(all_proteins)
    train_labels, val_labels, test_labels = split_train_val_test(all_labels)
    train_m_names, val_m_names, test_m_names = split_train_val_test(molecules_names)
    train_p_names, val_p_names, test_p_names = split_train_val_test(proteins_names)

    val_e_types = get_test_e_type(train_p_names, val_p_names, train_m_names, val_m_names)
    test_e_types = get_test_e_type(train_p_names, test_p_names, train_m_names, test_m_names)

    train_loader = data_to_loader(train_molecules, train_proteins, train_labels, None, batch_size=bs, shuffle=True)
    val_loader = data_to_loader(val_molecules, val_proteins, val_labels, val_e_types, batch_size=bs, shuffle=False)
    test_loader = data_to_loader(test_molecules, test_proteins, test_labels, test_e_types, batch_size=bs, shuffle=False)
    model = ProteinDrugLinearModel(fuse_base, m_fuse=m_fuse, p_fuse=p_fuse, m_model=m_model, p_model=p_model,
                                   only_rand=only_rand, fuse_freeze=fuse_freeze).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_val_auc = 0
    best_test_auc = 0
    best_train_all_auc = 0
    for epoch in range(100):
        train_auc = run_epoch(model, train_loader, optimizer, loss_func, "train")
        with torch.no_grad():
            val_auc = run_epoch(model, val_loader, optimizer, loss_func, "val")
            test_auc = run_epoch(model, test_loader, optimizer, loss_func, "test")
        print(
            f"Epoch {epoch}: train: {score_to_str(train_auc)}, val: {score_to_str(val_auc)} test {score_to_str(test_auc)}")
        best_train_all_auc = max(best_train_all_auc, train_auc[0])
        if val_auc[1] > best_val_auc:
            best_val_auc = val_auc[1]
            best_test_auc = test_auc[1]
    msg = f"{fuse_name},{model_to_conf_name(model)},{best_val_auc},{best_test_auc},{best_train_all_auc}"
    print(msg)
    with open("results.txt", "a") as f:
        f.write(msg + "\n")
