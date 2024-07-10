import os
from os.path import join as pjoin
import numpy as np
import torch
from torchdrug.metrics import area_under_roc
from torch.utils.data import Dataset, DataLoader

from common.data_types import MOLECULE
from common.path_manager import data_path, scores_path, model_path
from common.utils import get_type_to_vec_dim
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(mol_emd, task_name):
    base_dir = f"{data_path}/mol/"
    mol_file = pjoin(base_dir, f"{task_name}_{mol_emd}_molecules.npy")
    labels_output_file = pjoin(base_dir, f"{task_name}_label.npy")
    data = np.load(mol_file)[:, 0, :]
    labels = np.load(labels_output_file)
    return data, labels


def split_train_val_test(data, train_size=0.7, val_size=0.15):
    train_val_index = int(len(data) * train_size)
    val_test_index = train_val_index + int(len(data) * val_size)
    train_data = data[:train_val_index]
    val_data = data[train_val_index:val_test_index]
    test_data = data[val_test_index:]
    return train_data, val_data, test_data


class MolLabelDatast(Dataset):
    def __init__(self, mols, labels):
        self.mols = mols
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mols[idx], self.labels[idx]


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
    dataset = MolLabelDatast(proteins, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_layers(dims):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
    return layers


class MolLabelModel(torch.nn.Module):
    def __init__(self, fuse_base, mol_dim, use_fuse, use_model, output_dim=1, fuse_freeze=True, fuse_model=None):
        super().__init__()
        self.input_dim = 0
        self.use_fuse = use_fuse
        self.use_model = use_model
        self.fuse_freeze = fuse_freeze

        if self.use_fuse:
            if fuse_model is None:
                self.fuse_model, dim = load_fuse_model(fuse_base)
            else:
                self.fuse_model = fuse_model
                dim = fuse_model.output_dim[0]
            self.input_dim += dim
            self.m_type = "molecule_protein"
        if use_model:
            self.input_dim += mol_dim
        self.layers = get_layers([self.input_dim, 512, output_dim])

    def forward(self, protein):
        x = []
        if self.use_fuse:
            fuse_mol = self.fuse_model(protein, self.m_type)
            if self.fuse_freeze:
                fuse_mol = fuse_mol.detach()
            x.append(fuse_mol)
        if self.use_model:
            x.append(protein)
        x = torch.concat(x, dim=1)
        return self.layers(x)


def run_epoch(model, loader, optimizer, criterion, part):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    for mols, labels in loader:
        mols = mols.to(device).float()
        optimizer.zero_grad()
        labels = labels.float().to(device)
        output = model(mols)
        loss = criterion(output, labels)
        if part == "train":
            loss.backward()
            optimizer.step()
        output = torch.sigmoid(output).detach()
        reals.append(labels)
        preds.append(output)
    if part != "train":
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        score = area_under_roc(reals.flatten(), preds.flatten()).item()
        return score
    else:
        return 0


def model_to_conf_name(model: MolLabelModel):
    return f"{model.use_fuse},{model.use_model},"


def main(args, fuse_model=None):
    fuse_base = args.dp_fuse_base
    use_fuse = bool(args.cafa_use_fuse)
    use_model = bool(args.cafa_use_model)
    fuse_freeze = bool(args.cafa_fuse_freeze)
    bs = args.dp_bs
    lr = args.dp_lr
    mol_emd_type = args.mol_emd
    mol_task = args.mol_task

    seed = args.random_seed
    np.random.seed(seed)
    type_to_vec_dim = get_type_to_vec_dim()
    mols, labels = load_data(mol_emd_type, mol_task)
    shuffle_index = np.random.permutation(len(mols))
    mols = mols[shuffle_index]
    labels = labels[shuffle_index]
    train_proteins, val_proteins, test_proteins = split_train_val_test(mols)
    train_labels, val_labels, test_labels = split_train_val_test(labels)

    train_loader = data_to_loader(train_proteins, train_labels, batch_size=bs, shuffle=True)
    val_loader = data_to_loader(val_proteins, val_labels, batch_size=bs, shuffle=False)
    test_loader = data_to_loader(test_proteins, test_labels, batch_size=bs, shuffle=False)

    model = MolLabelModel(fuse_base, type_to_vec_dim[MOLECULE], use_fuse, use_model, labels.shape[1],
                          fuse_freeze, fuse_model=fuse_model).to(device)
    if args.dp_print:
        print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_weight = (1 + train_labels.sum(axis=0)) / train_labels.shape[0]
    pos_weight = (1 - pos_weight) / pos_weight
    pos_weight = torch.tensor(pos_weight, device=device, dtype=torch.float32)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_acc = 0
    best_test_acc = 0
    no_improve = 0
    for epoch in range(250):
        train_score = run_epoch(model, train_loader, optimizer, loss_func, "train")
        with torch.no_grad():
            val_score = run_epoch(model, val_loader, optimizer, loss_func, "val")
            test_score = run_epoch(model, test_loader, optimizer, loss_func, "test")

        if args.dp_print:
            # print(train_score.to_string())
            print(epoch, train_score, val_score, test_score)
        if val_score > best_val_acc:
            best_val_acc = val_score
            best_test_acc = test_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.max_no_improve:
                break

    if args.dp_print:
        print("Best Test scores\n", best_test_acc)
        output_file = f"{scores_path}/mol_{mol_task}.csv"
        if not os.path.exists(output_file):
            names = "name,use_fuse,use_model,"
            with open(output_file, "w") as f:
                f.write(names + "acc" + "\n")
        with open(output_file, "a") as f:
            f.write(f'{args.name},' + model_to_conf_name(model) + str(best_test_acc) + "\n")
    return best_val_acc, best_test_acc


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
