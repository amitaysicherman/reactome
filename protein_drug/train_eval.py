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


def load_fuse_model(base_dir, cp_name):
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


class ProteinDrugLinearModel(torch.nn.Module):
    def __init__(self, fuse_base, fuse_name, molecule_dim=768, protein_dim=1024, use_fuse=True, use_model=True):
        super().__init__()
        if use_fuse:
            self.fuse_model, dim = load_fuse_model(fuse_base, fuse_name)
            self.molecule_dim = dim
            self.protein_dim = dim
        else:
            self.molecule_dim = molecule_dim
            self.protein_dim = protein_dim
        self.use_fuse = use_fuse
        self.use_model = use_model

        self.fc_m = torch.nn.Linear(1024 + 768, 1024)
        self.fc_p = torch.nn.Linear(1024 + 1024, 1024)
        self.relu = torch.nn.ReLU()
        # self.fc_last = torch.nn.Linear(1024, 1)

    def forward(self, molecule, protein):

        if self.use_fuse:
            molecule_f = self.fuse_model(molecule, "molecule_protein").detach()
            protein_f = self.fuse_model(protein, "protein_protein").detach()
            molecule = torch.cat([molecule, molecule_f], dim=1)
            protein = torch.cat([protein, protein_f], dim=1)
            if not self.use_model:
                return -1 * F.cosine_similarity(molecule_f, protein_f).unsqueeze(1)
        molecule = self.fc_m(molecule)
        molecule = self.relu(molecule)
        protein = self.fc_p(protein)
        protein = self.relu(protein)
        return -1 * F.cosine_similarity(molecule, protein).unsqueeze(1)
        # x = m + p
        # x = self.fc_last(x)
        # return x


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


if __name__ == "__main__":
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

    train_loader = data_to_loader(train_molecules, train_proteins, train_labels, None, batch_size=32, shuffle=True)
    val_loader = data_to_loader(val_molecules, val_proteins, val_labels, val_e_types, batch_size=32, shuffle=False)
    test_loader = data_to_loader(test_molecules, test_proteins, test_labels, test_e_types, batch_size=32, shuffle=False)

    model = ProteinDrugLinearModel("data/models_checkpoints/fuse_all-to-prot", "fuse_47.pt", use_fuse=True,
                                   use_model=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.BCEWithLogitsLoss()
    for epoch in range(100):
        train_auc = run_epoch(model, train_loader, optimizer, loss_func, "train")
        with torch.no_grad():
            val_auc = run_epoch(model, val_loader, optimizer, loss_func, "val")
            test_auc = run_epoch(model, test_loader, optimizer, loss_func, "test")
        print(
            f"Epoch {epoch}: train: {score_to_str(train_auc)}, val: {score_to_str(val_auc)} test {score_to_str(test_auc)}")
