import os
from os.path import join as pjoin
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

from common.data_types import PROTEIN
from common.path_manager import data_path, scores_path, model_path
from common.utils import get_type_to_vec_dim
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(prot_emd_type, is_bin):
    if is_bin:
        dir = "subcellular_localization_2"
    else:
        dir = "subcellular_localization"
    loc_dir = "localization"
    protein_file = pjoin(data_path, loc_dir, f"{dir}_{prot_emd_type}.npy")
    target_file = pjoin(data_path, loc_dir, f"{dir}_target.npy")
    data = np.load(protein_file)[:, 0, :]
    labels = np.load(target_file)
    return data, labels


def split_train_val_test(data, train_size, val_size, test_size):
    assert len(data) == train_size + val_size + test_size, f"{len(data)} != {train_size} + {val_size} + {test_size}"
    train_val_index = train_size
    val_test_index = train_size + val_size
    train_data = data[:train_val_index]
    val_data = data[train_val_index:val_test_index]
    test_data = data[val_test_index:]
    return train_data, val_data, test_data


class ProteiLabelDataset(Dataset):
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
    dataset = ProteiLabelDataset(proteins, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_layers(dims):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
    return layers


class ProteinLabelModel(torch.nn.Module):
    def __init__(self, fuse_base, protein_dim, use_fuse, use_model, output_dim=10, fuse_freeze=True, fuse_model=None):
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

        self.layers = get_layers([self.input_dim, 512, output_dim])

    def forward(self, protein):
        x = []
        if self.use_fuse:
            fuse_protein = self.fuse_model(protein, self.p_type)
            if self.fuse_freeze:
                fuse_protein = fuse_protein.detach()
            x.append(fuse_protein)
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
        preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy().flatten().tolist())

    real, pred = np.array(reals), np.array(preds)
    acc = accuracy_score(real, pred)
    return acc


def model_to_conf_name(model: ProteinLabelModel):
    return f"{model.use_fuse},{model.use_model},"


def main(args, fuse_model=None):
    fuse_base = args.dp_fuse_base
    use_fuse = bool(args.cafa_use_fuse)
    use_model = bool(args.cafa_use_model)
    fuse_freeze = bool(args.cafa_fuse_freeze)
    bs = args.dp_bs
    lr = args.dp_lr
    is_bin = args.loc_bin
    prot_emd_type = args.protein_emd
    seed = args.random_seed
    np.random.seed(seed)
    type_to_vec_dim = get_type_to_vec_dim(prot_emd_type)
    proteins, labels = load_data(prot_emd_type, is_bin)
    # shuffle_index = np.random.permutation(len(proteins))
    # proteins = proteins[shuffle_index]
    # labels = labels[shuffle_index]
    if args.dp_print:
        print(proteins.shape, labels.shape)
    if is_bin:
        sizes = {"train_size": 5184, "val_size": 1729, "test_size": 1749}
    else:
        sizes = {"train_size": 8420, "val_size": 2811, "test_size": 2773}
    print(sizes)
    train_proteins, val_proteins, test_proteins = split_train_val_test(proteins, **sizes)
    train_labels, val_labels, test_labels = split_train_val_test(labels, **sizes)

    train_loader = data_to_loader(train_proteins, train_labels, batch_size=bs, shuffle=True)
    val_loader = data_to_loader(val_proteins, val_labels, batch_size=bs, shuffle=False)
    test_loader = data_to_loader(test_proteins, test_labels, batch_size=bs, shuffle=False)

    model = ProteinLabelModel(fuse_base, type_to_vec_dim[PROTEIN], use_fuse, use_model, 10 if not is_bin else 2,
                              fuse_freeze, fuse_model=fuse_model).to(device)
    if args.dp_print:
        print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_func = torch.nn.CrossEntropyLoss()

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
        task_output_prefix = args.task_output_prefix
        output_file = f"{scores_path}/{task_output_prefix}loc{is_bin}.csv"
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
