from common.path_manager import data_path, scores_path, model_path
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from common.utils import get_type_to_vec_dim

from common.data_types import MOLECULE, PROTEIN
import torch
from model.models import MultiModalLinearConfig, MiltyModalLinear
# from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, \
#     auc as area_under_curve

from torchdrug.metrics import area_under_prc

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# class Score:
#     def __init__(self, epoch: int, part: str, auc: float, accuracy: float, precision: float, recall: float,
#                  aupr: float):
#         self.epoch = epoch
#         self.part = part
#         self.auc = auc
#         self.accuracy = accuracy
#         self.precision = precision
#         self.recall = recall
#         self.aupr = aupr
#
#     def to_string(self):
#         return f"{self.epoch},{self.part},{self.auc},{self.accuracy},{self.precision},{self.recall},{self.aupr}"

@staticmethod
def get_header():
    return "epoch,part,auc,accuracy,precision,recall,aupr"


def load_data(dataset, prot_emd_type, mol_emd_type):
    base_dir = os.path.join(data_path, "protein_drug")
    molecules = np.load(os.path.join(base_dir, f"{dataset}_{mol_emd_type}_molecules.npy"))[:, 0, :]
    proteins = np.load(os.path.join(base_dir, f"{dataset}_{prot_emd_type}_proteins.npy"))[:, 0, :]
    with open(os.path.join(base_dir, f"{dataset}_{mol_emd_type}_molecules.txt")) as f:
        molecules_names = f.read().splitlines()
    with open(os.path.join(base_dir, f"{dataset}_{prot_emd_type}_proteins.txt")) as f:
        proteins_names = f.read().splitlines()
    with open(os.path.join(base_dir, f"{dataset}_labels.txt")) as f:
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
        if e_types is not None:
            self.labels = self.labels[e_types == 1]
            self.molecules = self.molecules[e_types == 1]
            self.proteins = self.proteins[e_types == 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.molecules[idx], self.proteins[idx], self.labels[idx]


def load_fuse_model(base_dir):
    base_dir = str(os.path.join(model_path, base_dir))
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
    def __init__(self, fuse_base, type_to_vec_dim, m_fuse=True, p_fuse=True,
                 m_model=True, p_model=True, fuse_freeze=True, trans_dim=256,
                 fuse_model=None):
        super().__init__()
        self.m_fuse = m_fuse
        self.p_fuse = p_fuse
        self.m_model = m_model
        self.p_model = p_model
        self.fuse_base = fuse_base
        self.fuse_freeze = fuse_freeze
        if m_fuse or p_fuse:
            if fuse_model is None:
                self.fuse_model, dim = load_fuse_model(fuse_base)
            else:
                self.fuse_model = fuse_model
                dim = fuse_model.output_dim[0]
            if m_fuse:
                self.m_fuse_linear = torch.nn.Linear(dim, trans_dim)
                if "molecule_protein" in self.fuse_model.names:
                    self.m_type = "molecule_protein"
                elif "molecule" in self.fuse_model.names:
                    self.m_type = "molecule"
                elif "molecule_molecule" in self.fuse_model.names:
                    self.m_type = "molecule_molecule"
                else:
                    raise ValueError("No molecule type in the model")
            if p_fuse:
                self.p_fuse_linear = torch.nn.Linear(dim, trans_dim)
                if "protein_protein" in self.fuse_model.names:
                    self.p_type = "protein_protein"
                elif "protein" in self.fuse_model.names:
                    self.p_type = "protein"
                elif "protein_molecule" in self.fuse_model.names:
                    self.p_type = "protein_molecule"
                else:
                    raise ValueError("No protein type in the model")

        if m_model:
            self.m_model_linear = torch.nn.Linear(type_to_vec_dim[MOLECULE], trans_dim)
        if p_model:
            self.p_model_linear = torch.nn.Linear(type_to_vec_dim[PROTEIN], trans_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=trans_dim, nhead=2, dim_feedforward=trans_dim * 2,
                                                         batch_first=True, dropout=0.5)
        self.trans = torch.nn.Sequential(
            torch.nn.TransformerEncoder(encoder_layer, num_layers=2),
            torch.nn.Linear(trans_dim, 1)
        )

    def forward(self, molecule, protein):
        x = []
        if self.m_fuse:
            fuse_molecule = self.fuse_model(molecule, self.m_type)
            if self.fuse_freeze:
                fuse_molecule = fuse_molecule.detach()
            fuse_molecule = self.m_fuse_linear(fuse_molecule)
            x.append(fuse_molecule)
        if self.m_model:
            molecule = self.m_model_linear(molecule)
            x.append(molecule)
        if self.p_fuse:
            fuse_protein = self.fuse_model(protein, self.p_type)
            if self.fuse_freeze:
                fuse_protein = fuse_protein.detach()
            fuse_protein = self.p_fuse_linear(fuse_protein)
            x.append(fuse_protein)
        if self.p_model:
            protein = self.p_model_linear(protein)
            x.append(protein)
        mol_protein = torch.stack(x, dim=1)
        return self.trans(mol_protein).mean(dim=1)


def run_epoch(model, loader, optimizer, criterion, part, epoch):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    total_loss = 0
    for molecules, proteins, labels in loader:
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
        reals.append(labels.unsqueeze(1).detach())
        preds.append(torch.sigmoid(outputs).detach())
    if part != "train":
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        score = area_under_prc(preds.flatten(), reals.flatten()).item()
        return score
    else:
        return 0


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
    test_types = np.array(test_types)
    return test_types


# def score_to_str(score_dict):
#     if 0 in score_dict:
#         return f'{score_dict[0] * 100:.2f}'
#     else:
#         return f'{score_dict[1] * 100:.2f}'
#     # return " ".join([f"{k}:{v * 100:.1f}" for k, v in score_dict.items()])
#

def model_to_conf_name(model):
    return f"{model.m_fuse},{model.p_fuse},{model.m_model},{model.p_model},"


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


def main(args, fuse_model=None):
    fuse_base = args.dp_fuse_base
    m_fuse = bool(args.dp_m_fuse)
    p_fuse = bool(args.dp_p_fuse)
    m_model = bool(args.dp_m_model)
    p_model = bool(args.dp_p_model)
    only_rand = bool(args.dp_only_rand)
    fuse_freeze = bool(args.dp_fuse_freeze)
    bs = args.dp_bs
    lr = args.dp_lr
    prot_emd_type = args.protein_emd
    mol_emd_type = args.mol_emd
    dataset = args.db_dataset
    seed = args.random_seed
    np.random.seed(seed)
    type_to_vec_dim = get_type_to_vec_dim(prot_emd_type)
    all_molecules, all_proteins, all_labels, molecules_names, proteins_names = load_data(dataset, prot_emd_type,
                                                                                         mol_emd_type)
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
    model = ProteinDrugLinearModel(fuse_base, type_to_vec_dim, m_fuse=m_fuse, p_fuse=p_fuse, m_model=m_model,
                                   p_model=p_model,
                                   fuse_freeze=fuse_freeze, fuse_model=fuse_model).to(device)
    if args.dp_print:
        print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    positive_sample_weight = train_labels.sum() / len(train_labels)
    negative_sample_weight = 1 - positive_sample_weight
    pos_weight = negative_sample_weight / positive_sample_weight
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    best_val_score = 0
    best_test_score = 0
    no_improve = 0
    for epoch in range(250):
        _ = run_epoch(model, train_loader, optimizer, loss_func, "train", epoch)
        with torch.no_grad():
            val_score = run_epoch(model, val_loader, optimizer, loss_func, "val", epoch)
            test_score = run_epoch(model, test_loader, optimizer, loss_func, "test", epoch)

        if args.dp_print:
            print(epoch, val_score, test_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_test_score = test_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.max_no_improve:
                break

    if args.dp_print:
        print("Best Test scores\n", best_test_score)
        output_file = f"{scores_path}/drug_protein_{dataset}.csv"
        if not os.path.exists(output_file):
            names = "name,m_fuse,p_fuse,m_model,p_model,"

            with open(output_file, "w") as f:
                f.write(names + f'auprc,' + "\n")
        with open(output_file, "a") as f:
            f.write(f'{args.name},' + model_to_conf_name(model) + str(best_test_score) + "\n")
    return best_val_score, best_test_score


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
