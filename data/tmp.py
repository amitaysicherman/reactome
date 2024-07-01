from torch.nn import Module
import pandas as pd
import os
import torch
from model.models import MultiModalLinearConfig, MiltyModalLinear
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

names_dict = {"BRD4": 0, "HSA": 1, "sEH": 2}

batch_size = 5000


class MolToVec:
    def __init__(self):
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").base_model.eval().to(
            device)
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    def to_vec(self, seqs: list):
        inputs = self.tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            # Pass both input_ids and attention_mask to the model
            hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]

        # Compute the mean of the hidden states, ignoring padding tokens
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden_states = hidden_states * attention_mask
        sum_hidden_states = masked_hidden_states.sum(dim=1)
        count_tokens = attention_mask.sum(dim=1)
        vecs = sum_hidden_states / count_tokens

        return self.post_process(vecs)

    def post_process(self, vecs):
        vecs = vecs.detach()  # .cpu().numpy()
        return vecs


def load_fuse_model(base_dir, cp_name):
    cp_data = torch.load(f"{base_dir}/{cp_name}", map_location=torch.device('cpu'))
    config_file = os.path.join(base_dir, 'config.txt')
    config = MultiModalLinearConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = MiltyModalLinear(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


class MoleculeProtModel(Module):
    def __init__(self, fuse_base, fuse_name, molecule_dim=768, n_layers=5, use_fuse=True):
        super().__init__()
        self.molecule_dim = molecule_dim
        self.mol_to_vec = MolToVec()
        self.lin_dim = 1024
        if use_fuse:
            self.fuse_model, dim = load_fuse_model(fuse_base, fuse_name)
            self.molecule_dim += dim
        else:
            self.molecule_dim = molecule_dim
        self.use_fuse = use_fuse
        self.relu = torch.nn.ReLU()
        self.fc_m = torch.nn.Linear(self.molecule_dim, self.lin_dim)
        self.lin_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.lin_layers.append(torch.nn.Linear(self.lin_dim, self.lin_dim))
        self.fc_last = torch.nn.Linear(self.lin_dim, 1)
        self.batch_norm = torch.nn.BatchNorm1d(self.lin_dim)

    def forward(self, molecule):
        molecule = self.mol_to_vec.to_vec(molecule)
        if self.use_fuse:
            molecule_f = self.fuse_model(molecule, "molecule_protein").detach()
            molecule = torch.cat([molecule, molecule_f], dim=1)
        molecule = self.fc_m(molecule)
        molecule = self.relu(molecule)
        molecule = self.batch_norm(molecule)
        for layer in self.lin_layers:
            molecule = layer(molecule)
            molecule = self.relu(molecule)
            molecule = self.batch_norm(molecule)
        x = self.fc_last(molecule)
        return x


class DataIterator:
    def __init__(self, batch_size, prot_index, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = 1e6
        self.prot_index = prot_index
        self.shuffle = shuffle
        self.curr_labels = None
        self.curr_vecs = None
        self.curr_prots = None

    def __iter__(self):
        for c_index, chunk in tqdm(enumerate(pd.read_csv('train.csv', chunksize=1e6))):
            if self.shuffle:
                chunk = chunk.sample(frac=1)
            proteins = chunk['protein_name'].apply(lambda x: names_dict[x]).values
            molecules = chunk['molecule_smiles'].values
            labels = chunk['binds'].values
            proteins_mask = proteins == self.prot_index
            molecules = molecules[proteins_mask]
            labels = labels[proteins_mask]

            for i in range(0, len(molecules), self.batch_size):
                yield molecules[i:i + self.batch_size].tolist(), labels[i:i + self.batch_size]
                if self.shuffle and i > 10:
                    break


def plot_and_print(reals, preds):
    print(roc_auc_score(reals, preds), average_precision_score(reals, preds))


def run_epoch(model, criterion, optimizer, data_iter, mode='train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    total_loss = 0
    for index, (smiles, labels) in enumerate(data_iter):
        # vecs = torch.tensor(vecs).to(device)
        labels = torch.tensor(labels).to(device).float()
        optimizer.zero_grad()
        outputs = model(smiles)
        loss = criterion(outputs, labels.unsqueeze(1))
        if mode == 'train':
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        reals.extend(labels.cpu().numpy())
        preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        if index % 100 == 0:
            plot_and_print(reals, preds)
    return total_loss, roc_auc_score(reals, preds), average_precision_score(reals, preds)


def load_lasts(model, optimizer, dir):
    if os.path.exists(f"{dir}/last.pt"):
        model.load_state_dict(torch.load(f"{dir}/last.pt"))
        optimizer.load_state_dict(torch.load(f"{dir}/last_opt.pt"))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bz', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_fuse', type=int, default=1)

args = parser.parse_args()

model = MoleculeProtModel("/home/amitay.s/reactome/data/models_checkpoints/fuse_all-to-prot", "fuse_47.pt",
                          use_fuse=args.use_fuse).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([150.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
load_lasts(model, optimizer, "cp")
data_iter = DataIterator(args.bz, 0)

os.makedirs("cp", exist_ok=True)
with open("results.txt", "w") as f:
    f.write("")

for epoch in range(100):
    train_loss, train_auc, train_ap = run_epoch(model, criterion, optimizer, data_iter, "train")
    print(f"Epoch {epoch} train_loss: {train_loss}, train_auc: {train_auc}, train_ap: {train_ap}")
    with open("results.txt", "a") as f:
        f.write(f"Epoch {epoch} train_loss: {train_loss}, train_auc: {train_auc}, train_ap: {train_ap}\n")
    os.remove("cp/last.pt")
    os.remove("cp/last_opt.pt")
    torch.save(model.state_dict(), f"cp/last.pt")
    torch.save(optimizer.state_dict(), f"cp/last_opt.pt")
