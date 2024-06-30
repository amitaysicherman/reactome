from common.path_manager import data_path
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from model.models import MultiModalLinearConfig, MiltyModalLinear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data():
    base_dir = os.path.join(data_path, "protein_drug")
    molecules = np.load(os.path.join(base_dir, "molecules.npy"))
    proteins = np.load(os.path.join(base_dir, "proteins.npy"))
    with open(os.path.join(base_dir, "labels.txt")) as f:
        labels = f.read().splitlines()
    labels = np.array(labels, dtype=np.float32)
    return molecules, proteins, labels


def split_train_val_test(data, val_size=0.16, test_size=0.20):
    train_val_index = int((1 - val_size - test_size) * len(molecules))
    val_test_index = int((1 - test_size) * len(molecules))
    train_data = data[:train_val_index]
    val_data = data[train_val_index:val_test_index]
    test_data = data[val_test_index:]
    return train_data, val_data, test_data


class ProteinDrugDataset(Dataset):
    def __init__(self, molecules, proteins, labels):
        self.molecules = molecules
        self.proteins = proteins
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.molecules[idx], self.proteins[idx], self.labels[idx]


def load_fuse_model(model_cp):
    cp_data = torch.load(model_cp, map_location=torch.device('cpu'))
    config_file = os.path.join(os.path.basename(model_cp), 'config.txt')
    config = MultiModalLinearConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = MiltyModalLinear(config)
    model.load_state_dict(cp_data)
    model.eval()
    return model, dim


def data_to_loader(molecules, proteins, labels, batch_size=32, shuffle=True):
    dataset = ProteinDrugDataset(molecules, proteins, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ProteinDrugLinearModel(torch.nn.Module):
    def __init__(self, fuse_cp, molecule_dim=768, protein_dim=1024, use_fuse=True, use_model=True):
        super().__init__()
        if use_fuse:
            self.fuse_model, dim = load_fuse_model(fuse_cp)
            self.molecule_dim = dim
            self.protein_dim = dim
        else:
            self.molecule_dim = molecule_dim
            self.protein_dim = protein_dim
        self.use_fuse = use_fuse
        self.use_model = use_model

        self.fc_m = torch.nn.Linear(molecule_dim, 128)
        self.fc_p = torch.nn.Linear(protein_dim, 128)
        self.relu = torch.nn.ReLU()
        self.fc_last = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, molecule, protein):
        if self.use_fuse:
            molecule = self.fuse_model(molecule, "molecule_protein")
            protein = self.fuse_model(protein, "protein_protein")
        if not self.use_model:
            return F.cosine_similarity(molecule, protein)

        m = self.fc_m(molecule)
        m = self.relu(m)
        p = self.fc_p(protein)
        p = self.relu(p)
        x = m + p
        x = self.fc_last(x)
        return x


def run_epoch(model, loader, optimizer, criterion, part):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    total_loss = 0
    for molecules, proteins, labels in loader:
        molecules = molecules.to(device)
        proteins = proteins.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(molecules, proteins)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        reals.extend(labels.cpu().numpy().tolist())
        preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().tolist())
    auc = roc_auc_score(reals, preds)
    print(f"{part} loss: {total_loss / len(loader)}")
    return auc


if __name__ == "__main__":
    all_molecules, all_proteins, all_labels = load_data()
    print(all_molecules.shape, all_proteins.shape, all_labels.shape)
    train_molecules, val_molecules, test_molecules = split_train_val_test(all_molecules)
    train_proteins, val_proteins, test_proteins = split_train_val_test(all_proteins)
    train_labels, val_labels, test_labels = split_train_val_test(all_labels)
    train_loader = data_to_loader(train_molecules, train_proteins, train_labels, shuffle=True)
    val_loader = data_to_loader(val_molecules, val_proteins, val_labels, shuffle=False)
    test_loader = data_to_loader(test_molecules, test_proteins, test_labels, shuffle=False)

    model = ProteinDrugLinearModel("data/models_checkpoints/fuse_all-to-prot/fuse_47.pt").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.BCEWithLogitsLoss()
    for epoch in range(10):
        train_auc = run_epoch(model, train_loader, optimizer, loss_func, "train")
        with torch.no_grad():
            val_auc = run_epoch(model, val_loader, optimizer, loss_func, "val")
            test_auc = run_epoch(model, test_loader, optimizer, loss_func, "test")
        print(f"Epoch {epoch}: train_auc: {train_auc}, val_auc: {val_auc}")
