import sys
import os
from common.path_manager import data_path, scores_path
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from model.models import MultiModalLinearConfig, MiltyModalLinear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, \
    auc as area_under_curve

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


rand_memories = {}


def entity_to_rand(entity, dim):
    if entity in rand_memories:
        return rand_memories[entity]
    rand_memories[entity] = np.random.randn(dim)
    return rand_memories[entity]


class ProteinDrugDataset(Dataset):
    def __init__(self, molecules, proteins, labels, e_types=None, only_rand=False, molecules_names=None,
                 proteins_names=None):
        self.molecules = molecules
        if only_rand:
            self.molecules = np.stack([entity_to_rand(m, self.molecules.shape[1]) for m in molecules_names])
        self.proteins = proteins
        if only_rand:
            self.proteins = np.stack([entity_to_rand(p, self.proteins.shape[1]) for p in proteins_names])
        self.labels = labels
        self.e_types = e_types if e_types is not None else [0] * len(molecules)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.molecules[idx], self.proteins[idx], self.labels[idx], self.e_types[idx]


def load_fuse_model(base_dir):
    cp_names = os.listdir(base_dir)
    cp_name = [x for x in cp_names if x.endswith(".pt")][0]
    print(f"Load model {base_dir}/{cp_name}")
    cp_data = torch.load(f"{base_dir}/{cp_name}", map_location=torch.device('cpu'))
    config_file = os.path.join(base_dir, 'config.txt')
    config = MultiModalLinearConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = MiltyModalLinear(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


def data_to_loader(molecules, proteins, labels, e_types, batch_size, shuffle=True, only_rand=False,
                   molecules_names=None,
                   proteins_names=None):
    dataset = ProteinDrugDataset(molecules, proteins, labels, e_types, only_rand, molecules_names, proteins_names)
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
                 m_model=True, p_model=True, only_rand=False, fuse_freeze=True, use_transformer=True, trans_dim=256):
        super().__init__()
        self.molecule_dim = 0
        self.protein_dim = 0
        self.m_fuse = m_fuse
        self.p_fuse = p_fuse
        self.m_model = m_model
        self.p_model = p_model
        self.fuse_base = fuse_base
        self.fuse_freeze = fuse_freeze
        if m_fuse or p_fuse:
            self.fuse_model, dim = load_fuse_model(fuse_base)
            if m_fuse:
                if "molecule_protein" in self.fuse_model.names:
                    self.m_type = "molecule_protein"
                elif "molecule" in self.fuse_model.names:
                    self.m_type = "molecule"
                elif "molecule_molecule" in self.fuse_model.names:
                    self.m_type = "molecule_molecule"
                else:
                    raise ValueError("No molecule type in the model")

                self.molecule_dim += dim
            if p_fuse:
                if "protein_protein" in self.fuse_model.names:
                    self.p_type = "protein_protein"
                elif "protein" in self.fuse_model.names:
                    self.p_type = "protein"
                elif "protein_molecule" in self.fuse_model.names:
                    self.p_type = "protein_molecule"
                else:
                    raise ValueError("No protein type in the model")

                self.protein_dim += dim

        if m_model:
            self.molecule_dim += MOL_DIM
        if p_model:
            self.protein_dim += PROT_DIM
        self.only_rand = only_rand
        self.use_transformer = use_transformer
        if not self.use_transformer:
            self.m_layers = get_layers([self.molecule_dim, 1024, 512, 256, 128])
            self.p_layers = get_layers([self.protein_dim, 1024, 512, 256, 128])
        else:
            self.m_layers = get_layers([self.molecule_dim, trans_dim])
            self.p_layers = get_layers([self.protein_dim, trans_dim])
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=trans_dim, nhead=2, dim_feedforward=trans_dim * 2,
                                                             batch_first=True)
            self.trans = torch.nn.Sequential(
                torch.nn.TransformerEncoder(encoder_layer, num_layers=2),
                torch.nn.Linear(trans_dim, 1)
            )

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
        if self.use_transformer:
            molecule = self.m_layers(molecule)
            protein = self.p_layers(protein)
            mol_protein = torch.stack((molecule, protein), dim=1).mean(dim=1)

            return self.trans(mol_protein)
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
        molecules = molecules.to(device).float()
        proteins = proteins.to(device).float()
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
        real, pred = np.array(reals[key]), np.array(preds[key])
        auc[key] = roc_auc_score(real, pred)
        print(f"{part} {key} AUC: {auc[key]}")
        print(f"{part} {key} accuracy: {accuracy_score(real, pred > 0.5)}")
        print(f"{part} {key} precision: {precision_score(real, pred > 0.5)}")
        print(f"{part} {key} recall: {recall_score(real, pred > 0.5)}")
        print(f"{part} {key} precision_recall_curve: {precision_recall_curve(real, pred)}")
        precision, recall, thresholds = precision_recall_curve(real, pred)
        print(f"{part} {key} auc: {area_under_curve(recall, precision)}")

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
    fuse_name = model.fuse_base.split("/")[-1]
    return f"{model.m_fuse},{model.p_fuse},{model.m_model},{model.p_model},{fuse_name}"


def get_all_args_opt():
    conf = []

    for m_fuse in [0, 1]:
        for p_fuse in [0, 1]:
            for m_model in [0, 1]:
                for p_model in [0, 1]:
                    for fuse in ["fuse_all-to-prot", "fuse_recon", "fuse_all-to-all", "fuse_inv", "fuse_fuse"]:
                        if not m_fuse and not m_model:
                            continue
                        if not p_fuse and not p_model:
                            continue
                        conf.append(
                            f"--m_fuse {m_fuse} --p_fuse {p_fuse} --m_model {m_model} --p_model {p_model} --fuse_base data/models_checkpoints/{fuse}")
    return conf


def main(args):
    fuse_base = args.dp_fuse_base
    m_fuse = bool(args.dp_m_fuse)
    p_fuse = bool(args.dp_p_fuse)
    m_model = bool(args.dp_m_model)
    p_model = bool(args.dp_p_model)
    only_rand = bool(args.dp_only_rand)
    fuse_freeze = bool(args.dp_fuse_freeze)
    bs = args.dp_bs
    lr = args.dp_lr

    np.random.seed(42)
    all_molecules, all_proteins, all_labels, molecules_names, proteins_names = load_data()
    shuffle_index = np.random.permutation(len(all_molecules))
    all_molecules = all_molecules[shuffle_index]
    all_proteins = all_proteins[shuffle_index]
    all_labels = all_labels[shuffle_index]
    molecules_names = molecules_names[shuffle_index]
    proteins_names = proteins_names[shuffle_index]
    if args.dp_print:
        print(all_molecules.shape, all_proteins.shape, all_labels.shape)
    train_molecules, val_molecules, test_molecules = split_train_val_test(all_molecules)
    train_proteins, val_proteins, test_proteins = split_train_val_test(all_proteins)
    train_labels, val_labels, test_labels = split_train_val_test(all_labels)
    train_molecules_names, val_molecules_names, test_molecules_names = split_train_val_test(molecules_names)
    train_proteins_names, val_proteins_names, test_proteins_names = split_train_val_test(proteins_names)

    train_m_names, val_m_names, test_m_names = split_train_val_test(molecules_names)
    train_p_names, val_p_names, test_p_names = split_train_val_test(proteins_names)

    val_e_types = get_test_e_type(train_p_names, val_p_names, train_m_names, val_m_names)
    test_e_types = get_test_e_type(train_p_names, test_p_names, train_m_names, test_m_names)

    train_loader = data_to_loader(train_molecules, train_proteins, train_labels, None, batch_size=bs, shuffle=True,
                                  only_rand=only_rand, molecules_names=train_molecules_names,
                                  proteins_names=train_proteins_names)

    val_loader = data_to_loader(val_molecules, val_proteins, val_labels, val_e_types, batch_size=bs, shuffle=False,
                                only_rand=only_rand, molecules_names=val_molecules_names,
                                proteins_names=val_proteins_names)
    test_loader = data_to_loader(test_molecules, test_proteins, test_labels, test_e_types, batch_size=bs, shuffle=False,
                                 only_rand=only_rand, molecules_names=test_molecules_names,
                                 proteins_names=test_proteins_names)
    model = ProteinDrugLinearModel(fuse_base, m_fuse=m_fuse, p_fuse=p_fuse, m_model=m_model, p_model=p_model,
                                   only_rand=only_rand, fuse_freeze=fuse_freeze).to(device)
    if args.dp_print:
        print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_val_auc = 0
    best_test_auc = 0
    best_train_all_auc = 0
    no_improve = 0
    for epoch in range(100):
        train_auc = run_epoch(model, train_loader, optimizer, loss_func, "train")
        with torch.no_grad():
            val_auc = run_epoch(model, val_loader, optimizer, loss_func, "val")
            test_auc = run_epoch(model, test_loader, optimizer, loss_func, "test")
        if args.dp_print:
            print(
                f"Epoch {epoch}: train: {score_to_str(train_auc)}, val: {score_to_str(val_auc)} test {score_to_str(test_auc)}")
        best_train_all_auc = max(best_train_all_auc, train_auc[0])
        if val_auc[1] > best_val_auc:
            best_val_auc = val_auc[1]
            best_test_auc = test_auc[1]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.max_no_improve:
                break
    msg = f"{model_to_conf_name(model)},{best_val_auc},{best_test_auc},{best_train_all_auc}"
    if args.dp_print:
        print(msg)
        with open(f"{scores_path}/drug_bank.txt", "a") as f:
            f.write(msg + "\n")
    return best_val_auc, best_test_auc


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
