import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
from common.data_types import REAL, PROTEIN, TEXT, MOLECULE
from common.data_types import Reaction
from common.utils import prepare_files
from dataset.dataset_builder import get_reactions, add_if_not_none
from dataset.index_manger import NodesIndexManager, NodeData, get_from_args
from model.models import MultiModalSeq
from common.path_manager import scores_path
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)


class EpochScores:
    def __init__(self, name, output_file=""):
        self.name = name
        self.output_file = output_file
        self.loss = []
        self.y_real = []
        self.y_pred = []
        self.augmentations = []

    def add(self, output, y, loss, augmentations):
        self.loss.append(loss.item())
        self.y_real.extend(y.detach().cpu().numpy().tolist())
        self.y_pred.extend(torch.sigmoid(output).detach().cpu().numpy().tolist())
        self.augmentations.extend(augmentations)

    def calculate_auc(self):
        aug_type = np.array(self.augmentations)
        y_pred = np.array(self.y_pred)
        real_preds = np.array(y_pred[aug_type == REAL])

        real = [0] * len(real_preds)
        prot_preds = np.array(y_pred[aug_type == PROTEIN])
        prot_auc = roc_auc_score([1] * len(prot_preds) + real, np.concatenate([prot_preds, real_preds]))

        mol_preds = np.array(y_pred[aug_type == MOLECULE])
        mol_auc = roc_auc_score([1] * len(mol_preds) + real, np.concatenate([mol_preds, real_preds]))
        text_preds = np.array(y_pred[aug_type == TEXT])
        text_auc = roc_auc_score([1] * len(text_preds) + real, np.concatenate([text_preds, real_preds]))

        all_auc = roc_auc_score(self.y_real, self.y_pred)
        return all_auc, prot_auc, mol_auc, text_auc

    def log(self, i):
        # auc = roc_auc_score(self.y_real, self.y_pred)
        all_auc, prot_auc, mol_auc, text_auc = self.calculate_auc()
        loss = sum(self.loss) / len(self.loss)
        msg = f'{i},{self.name},{loss},{all_auc},{prot_auc},{mol_auc},{text_auc}'
        print(msg)
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(msg + "\n")
        return all_auc


def get_empty_dict():
    return {PROTEIN: [], MOLECULE: [], TEXT: []}


def reaction_to_nodes(reaction: Reaction, node_index_manager: NodesIndexManager):
    catalysis_activity_nodes = [node_index_manager.name_to_node[f'GO@{c.activity}'] for c in reaction.catalysis]
    catalysis_enetities_nodes = [node_index_manager.name_to_node[e.get_db_identifier()] for c in reaction.catalysis for
                                 e in c.entities]
    input_nodes = [node_index_manager.name_to_node[e.get_db_identifier()] for e in reaction.inputs]
    reaction_nodes = catalysis_activity_nodes + catalysis_enetities_nodes + input_nodes
    return reaction_nodes


class ReactionSeq:
    def __init__(self, nodes: List[NodeData], bp=None):
        self.data = get_empty_dict()
        for node in nodes:
            self.data[node.type].append(node)
        self.aug = REAL
        self.bp = bp

    def get_copy(self):
        copy_data = ReactionSeq([])
        for type_ in self.data:
            copy_data.data[type_] = self.data[type_][:]
        copy_data.aug = self.aug
        copy_data.bp = self.bp
        return copy_data

    def get_augment_copy(self, node_index_manager: NodesIndexManager, type=""):
        if type == "":
            n = sum(len(self.data[type_]) for type_ in self.data)
            p = [len(self.data[PROTEIN]) / n, len(self.data[MOLECULE]) / n, len(self.data[TEXT]) / n]

            type = random.choices([PROTEIN, MOLECULE, TEXT], weights=p)[0]

        if type not in self.data or len(self.data[type]) == 0:
            return None
        fake_index = random.choice(range(len(self.data[type])))
        entity_index = self.data[type][fake_index].index
        new_index = node_index_manager.sample_entity(entity_index, "random", type)
        copy_data = self.get_copy()
        copy_data.data[type][fake_index] = node_index_manager.index_to_node[new_index]
        copy_data.aug = type
        return copy_data

    def to_vecs_types(self):
        vec_types = get_empty_dict()
        for type_ in self.data:
            vec_types[type_] = [torch.Tensor(node.vec).float() for node in self.data[type_]]
        return vec_types.items()


def collate_fn(reaction_seqs: List[ReactionSeq], type_to_vec_dim, return_bp=False):
    data_by_type = get_empty_dict()
    labels = []
    augmentations = []
    bp_list = []
    for reaction_seq in reaction_seqs:
        labels.append(reaction_seq.aug != REAL)
        augmentations.append(reaction_seq.aug)
        bp_list.append(reaction_seq.bp)
        for type_, vecs in reaction_seq.to_vecs_types():
            data_by_type[type_].append(vecs)
    batch_data = {}
    batch_mask = {}
    for type_, vecs in data_by_type.items():
        max_len = max(len(v) for v in vecs)
        padded_vecs = torch.zeros(len(vecs), max_len, type_to_vec_dim[type_])
        mask = torch.zeros(len(vecs), max_len)
        for i, vec_list in enumerate(vecs):
            mask[i, :len(vec_list)] = 1
            for j, vec in enumerate(vec_list):
                padded_vecs[i, j, :] = vec

        batch_data[type_] = padded_vecs
        batch_mask[type_] = mask
    if return_bp:
        return batch_data, batch_mask, bp_list
    return batch_data, batch_mask, torch.tensor(labels), augmentations


def lines_to_dataset(lines, node_index_manager: NodesIndexManager, batch_size, shuffle, aug_factor, type_to_vec_dim):
    dataset = []
    for reaction in tqdm(lines):
        nodes = reaction_to_nodes(reaction, node_index_manager)
        data = ReactionSeq(nodes)
        dataset.append(data)
        for _ in range(aug_factor):
            add_if_not_none(dataset, data.get_augment_copy(node_index_manager))
    if shuffle:
        random.shuffle(dataset)
    all_data = []

    for i in range(0, len(dataset), batch_size):
        all_data.append(collate_fn(dataset[i:i + batch_size], type_to_vec_dim))
    if len(dataset) % batch_size != 0:
        all_data.append(collate_fn(dataset[-(len(dataset) % batch_size):], type_to_vec_dim))
    return all_data


def run_epoch(model, optimizer, loss_fn, dataset, part, output_file=""):
    is_train = part == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    score = EpochScores(part, output_file)

    for batch_data, batch_mask, labels, augmentations in dataset:
        optimizer.zero_grad()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        batch_mask = {k: v.to(device) for k, v in batch_mask.items()}
        output = model(batch_data, batch_mask)
        loss = loss_fn(output, labels.float().unsqueeze(-1).to(device))
        score.add(output, labels, loss, augmentations)
        if is_train:
            loss.backward()
            optimizer.step()
    auc = score.log(epoch)
    return auc


def print_best_results(results_file):
    columns = ['all', 'protein', 'molecule', 'text']
    with open(results_file, "r") as f:
        lines = f.readlines()
    train_results = pd.DataFrame(columns=columns)
    valid_results = pd.DataFrame(columns=columns)
    test_results = pd.DataFrame(columns=columns)
    for i in range(0, len(lines), 3):  # num,train,num,valid,num,test
        train_results.loc[i // 3] = [float(x) for x in lines[i].split(",")[3:]]
        valid_results.loc[i // 3] = [float(x) for x in lines[i + 1].split(",")[3:]]
        test_results.loc[i // 3] = [float(x) for x in lines[i + 2].split(",")[3:]]
    print("Best Results")
    print("Train results")
    print(train_results.max())
    print("Valid results")
    print(valid_results.max())
    print("Test results")
    print(test_results.max())

    # choose the best index for each column based on the valid results
    best_index = valid_results.idxmax()
    for col in valid_results.columns:
        print(f"Best model for {col}")
        print(test_results.loc[best_index[col]])
    name = os.path.basename(results_file).replace(".txt", "")
    summary = [name] + list(test_results.loc[best_index['all']].values)
    summary = ",".join([str(x) for x in summary])
    output_summary_file = f"{scores_path}/summary_seq.csv"
    if not os.path.exists(output_summary_file):
        with open(output_summary_file, "w") as f:
            f.write(",".join(["name"] + list(test_results.columns)) + "\n")
    with open(output_summary_file, "a") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    from common.args_manager import get_args

    batch_size = 2048
    lr = 0.001
    aug_factor = 5

    args = get_args()
    node_index_manager: NodesIndexManager = get_from_args(args)

    TYPE_TO_VEC_DIM = {PROTEIN: node_index_manager.index_to_node[node_index_manager.protein_indexes[0]].vec.shape[0],
                       MOLECULE: node_index_manager.index_to_node[node_index_manager.molecule_indexes[0]].vec.shape[0],
                       TEXT: node_index_manager.index_to_node[node_index_manager.text_indexes[0]].vec.shape[0]}

    filter_untrain = False
    if args.gnn_pretrained_method == 0:
        filter_untrain = True
    if "no-emb" in args.fuse_name:
        filter_untrain = True

    train_lines, val_lines, test_lines = get_reactions(args.gnn_sample, filter_untrain=filter_untrain, filter_dna=True,
                                                       filter_no_act=True)
    train_dataset = lines_to_dataset(train_lines, node_index_manager, batch_size, shuffle=True, aug_factor=aug_factor,
                                     type_to_vec_dim=TYPE_TO_VEC_DIM)
    val_dataset = lines_to_dataset(val_lines, node_index_manager, batch_size, shuffle=False, aug_factor=aug_factor,
                                   type_to_vec_dim=TYPE_TO_VEC_DIM)
    test_dataset = lines_to_dataset(test_lines, node_index_manager, batch_size, shuffle=False, aug_factor=aug_factor,
                                    type_to_vec_dim=TYPE_TO_VEC_DIM)
    print(len(train_lines), len(train_dataset))

    model = MultiModalSeq(args.seq_size, TYPE_TO_VEC_DIM, use_trans=args.seq_use_trans).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1 / aug_factor]).to(device))
    save_dir, score_file = prepare_files(f'seq_{args.name}')

    best_score = 0
    best_prev_index = -1
    for epoch in range(args.gnn_epochs):
        run_epoch(model, optimizer, loss_fn, train_dataset, "train", score_file)
        epoch_score = run_epoch(model, optimizer, loss_fn, val_dataset, "valid", score_file)
        run_epoch(model, optimizer, loss_fn, test_dataset, "test", score_file)
        if epoch_score > best_score:
            best_score = epoch_score
            torch.save(model.state_dict(), f"{save_dir}/{epoch}.pt")
            if best_prev_index != -1:
                os.remove(f"{save_dir}/{best_prev_index}.pt")
            best_prev_index = epoch
    print_best_results(score_file)
