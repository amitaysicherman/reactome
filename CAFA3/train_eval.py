import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, \
    auc as area_under_curve
from torch.utils.data import Dataset, DataLoader

from common.data_types import MOLECULE, PROTEIN
from common.path_manager import data_path, scores_path, model_path
from common.utils import get_type_to_vec_dim
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Score:
    def __init__(self, epoch: int, part: str, auc: float, accuracy: float, precision: float, recall: float,
                 aupr: float):
        self.epoch = epoch
        self.part = part
        self.auc = auc
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.aupr = aupr

    def to_string(self):
        return f"{self.epoch},{self.part},{self.auc},{self.accuracy},{self.precision},{self.recall},{self.aupr}"

    @staticmethod
    def get_header():
        return "epoch,part,auc,accuracy,precision,recall,aupr"


def load_data(task, prot_emd_type):
    train_proteins = np.load(f"{data_path}/CAFA3/preprocessed/train_protein_{task}_{prot_emd_type}.npy")
    test_proteins = np.load(f"{data_path}/CAFA3/preprocessed/test_protein_{task}_{prot_emd_type}.npy")
    with open(f"{data_path}/CAFA3/preprocessed/train_label_{task}_{prot_emd_type}.txt") as f:
        train_labels = f.read().split("\n")
    with open(f"{data_path}/CAFA3/preprocessed/test_label_{task}_{prot_emd_type}.txt") as f:
        test_labels = f.read().split("\n")
    train_labels = [[int(i) for i in l.split()] for l in train_labels]
    test_labels = [[int(i) for i in l.split()] for l in test_labels]
    max_label = max([max(l) for l in train_labels + test_labels])

    train_labels = np.zeros((len(train_labels), max_label + 1))
    for i, l in enumerate(train_labels):
        for j in l:
            train_labels[i, j] = 1
    test_labels = np.zeros((len(test_labels), max_label + 1))
    for i, l in enumerate(test_labels):
        for j in l:
            test_labels[i, j] = 1
    return train_proteins, test_proteins, train_labels, test_labels


def split_train_val(data, val_size=0.16):
    train_val_index = int((1 - val_size) * len(data))
    train_data = data[:train_val_index]
    val_data = data[train_val_index:]
    return train_data, val_data


class ProteinMultiLabelDataset(Dataset):
    def __init__(self, proteins, labels):
        self.proteins = proteins
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.proteins[idx], self.labels[idx]


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


def data_to_loader(proteins, labels, batch_size, shuffle=True):
    dataset = ProteinMultiLabelDataset(proteins, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_layers(dims):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
    return layers


class ProteinMiltilabelModel(torch.nn.Module):
    def __init__(self, fuse_base, protein_dim, use_fuse, use_model, fuse_freeze=True):
        super().__init__()
        self.input_dim = 0
        self.use_fuse = use_fuse
        self.use_model = use_model
        self.fuse_freeze = fuse_freeze

        if self.use_fuse:
            self.fuse_model, dim = load_fuse_model(fuse_base)
            self.input_dim += dim
            if "protein_protein" in self.fuse_model.names:
                self.p_type = "protein_protein"
            elif "protein" in self.fuse_model.names:
                self.p_type = "protein"
            elif "protein_molecule" in self.fuse_model.names:
                self.p_type = "protein_molecule"
            else:
                raise ValueError("No protein type in the model")
        if use_model:
            self.input_dim += protein_dim

        self.layers = get_layers([self.input_dim, 512, 128, 1])

    def forward(self, protein):
        x = []
        if self.use_fuse:
            fuse_protein = self.fuse_model(protein, self.p_type)
            if self.fuse_freeze:
                fuse_protein = fuse_protein.detach()
            fuse_protein = self.p_fuse_linear(fuse_protein)
            x.append(fuse_protein)
        if self.use_model:
            protein = self.p_model_linear(protein)
            x.append(protein)
        mol_protein = torch.stack(x, dim=1)
        return self.layers(mol_protein)


def run_epoch(model, loader, optimizer, criterion, part, epoch):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    total_loss = 0
    for proteins, labels in loader:
        proteins = proteins.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(proteins)
        loss = criterion(outputs, labels)
        if part == "train":
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        reals.extend(labels.detach().cpu().numpy().flatten().tolist())
        preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten().tolist())

    real, pred = np.array(reals), np.array(preds)
    precision, recall, thresholds = precision_recall_curve(real, pred)
    score = Score(auc=roc_auc_score(real, pred), epoch=epoch, part=part, accuracy=accuracy_score(real, pred > 0.5),
                  precision=precision_score(real, pred > 0.5), recall=recall_score(real, pred > 0.5),
                  aupr=area_under_curve(recall, precision))

    return score


def model_to_conf_name(model: ProteinMiltilabelModel):
    return f"{model.use_fuse},{model.use_model},"


def main(args):
    fuse_base = args.dp_fuse_base
    use_fuse = bool(args.cafa_use_fuse)
    use_model = bool(args.cafa_use_model)
    fuse_freeze = bool(args.cafa_fuse_freeze)
    task = args.cafa_task
    bs = args.dp_bs
    lr = args.dp_lr
    prot_emd_type = args.protein_emd
    seed = args.random_seed
    np.random.seed(seed)
    type_to_vec_dim = get_type_to_vec_dim(prot_emd_type)
    train_proteins, test_proteins, train_labels, test_labels = load_data(task, prot_emd_type)
    shuffle_index = np.random.permutation(len(train_proteins))
    train_proteins = train_proteins[shuffle_index]
    train_labels = train_labels[shuffle_index]
    if args.dp_print:
        print(train_proteins.shape, train_labels.shape, train_labels.shape)
    train_proteins, val_proteins = split_train_val(train_proteins)
    train_labels, val_labels = split_train_val(train_labels)

    train_loader = data_to_loader(train_proteins, train_labels, batch_size=bs, shuffle=True)
    val_loader = data_to_loader(val_proteins, val_labels, batch_size=bs, shuffle=False)
    test_loader = data_to_loader(test_proteins, test_labels, batch_size=bs, shuffle=False)

    model = ProteinMiltilabelModel(fuse_base, type_to_vec_dim[PROTEIN], use_fuse, use_model, fuse_freeze).to(device)
    if args.dp_print:
        print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    positive_sample_weight = train_labels.sum() / len(train_labels)
    negative_sample_weight = 1 - positive_sample_weight
    pos_weight = negative_sample_weight / positive_sample_weight
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

    best_val_auc = 0
    best_test_auc = 0
    best_test_score = None
    best_train_all_auc = 0
    no_improve = 0
    for epoch in range(250):
        train_score = run_epoch(model, train_loader, optimizer, loss_func, "train", epoch)
        with torch.no_grad():
            val_score = run_epoch(model, val_loader, optimizer, loss_func, "val", epoch)
            test_score = run_epoch(model, test_loader, optimizer, loss_func, "test", epoch)

        if args.dp_print:
            print(train_score.to_string())
            print(val_score.to_string())
            print(test_score.to_string())
        best_train_all_auc = max(best_train_all_auc, train_score.auc)
        if val_score.auc > best_val_auc:
            best_val_auc = val_score.auc
            best_test_auc = test_score.auc
            best_test_score = test_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.max_no_improve:
                break

    if args.dp_print:
        print("Best Test scores\n", best_test_score.to_string())
        output_file = f"{scores_path}/cafa_{task}.csv"
        if not os.path.exists(output_file):
            names = "name,m_fuse,p_fuse,m_model,p_model,"
            with open(output_file, "w") as f:
                f.write(names + Score.get_header() + "\n")
        with open(output_file, "a") as f:
            f.write(f'{args.name},' + model_to_conf_name(model) + best_test_score.to_string() + "\n")
    return best_val_auc, best_test_auc


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
