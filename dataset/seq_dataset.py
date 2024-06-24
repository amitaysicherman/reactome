import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from common.data_types import REAL, PROTEIN, TEXT, MOLECULE, TYPE_TO_VEC_DIM
from common.data_types import Reaction
from common.utils import prepare_files
from dataset.dataset_builder import get_reactions, add_if_not_none
from dataset.index_manger import NodesIndexManager, NodeData, get_from_args
from model.models import MultiModalSeq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        msg = f"{i} {self.name} loss: {sum(self.loss) / len(self.loss):.2f} all_auc: {all_auc * 100:.1f} prot_auc: {prot_auc * 100:.1f} mol_auc: {mol_auc * 100:.1f} text_auc: {text_auc * 100:.1f}"
        print(msg)
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(msg)


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
    def __init__(self, nodes: List[NodeData]):
        self.data = get_empty_dict()
        for node in nodes:
            self.data[node.type].append(node)
        self.aug = REAL

    def get_copy(self):
        copy_data = ReactionSeq([])
        for type_ in self.data:
            copy_data.data[type_] = self.data[type_][:]
        return copy_data

    def get_augment_copy(self, node_index_manager: NodesIndexManager, type):
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


def collate_fn(reaction_seqs: List[ReactionSeq]):
    data_by_type = get_empty_dict()
    labels = []
    augmentations = []

    for reaction_seq in reaction_seqs:
        labels.append(reaction_seq.aug != REAL)
        augmentations.append(reaction_seq.aug)
        for type_, vecs in reaction_seq.to_vecs_types():
            data_by_type[type_].append(vecs)
    batch_data = {}
    batch_mask = {}
    for type_, vecs in data_by_type.items():
        max_len = max(len(v) for v in vecs)
        padded_vecs = torch.zeros(len(vecs), max_len, TYPE_TO_VEC_DIM[type_])
        mask = torch.zeros(len(vecs), max_len)
        for i, vec_list in enumerate(vecs):
            mask[i, :len(vec_list)] = 1
            for j, vec in enumerate(vec_list):
                padded_vecs[i, j, :] = vec

        batch_data[type_] = padded_vecs
        batch_mask[type_] = mask
    return batch_data, batch_mask, torch.tensor(labels), augmentations


def lines_to_dataset(lines, node_index_manager: NodesIndexManager, batch_size, shuffle, protein_aug, molecule_aug,
                     text_aug):
    dataset = []
    for reaction in tqdm(lines):
        nodes = reaction_to_nodes(reaction, node_index_manager)
        data = ReactionSeq(nodes)
        dataset.append(data)
        for _ in range(protein_aug):
            add_if_not_none(dataset, data.get_augment_copy(node_index_manager, PROTEIN))
        for _ in range(molecule_aug):
            add_if_not_none(dataset, data.get_augment_copy(node_index_manager, MOLECULE))
        for _ in range(text_aug):
            add_if_not_none(dataset, data.get_augment_copy(node_index_manager, TEXT))
    if shuffle:
        random.shuffle(dataset)
    all_data = []
    for i in range(0, len(dataset), batch_size):
        all_data.append(collate_fn(dataset[i:i + batch_size]))
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
        loss = loss_fn(output, labels.float().unsqueeze(-1))
        score.add(output, labels, loss, augmentations)
        if is_train:
            loss.backward()
            optimizer.step()
    score.log(epoch)


if __name__ == "__main__":
    from common.args_manager import get_args

    batch_size = 2048
    emb_dim = 512
    lr = 0.001
    protein_aug = 5
    molecule_aug = 1
    text_aug = 1
    epochs = 100

    args = get_args()
    node_index_manager: NodesIndexManager = get_from_args(args)
    train_lines, val_lines, test_lines = get_reactions(args.gnn_sample, filter_untrain=False, filter_dna=True,
                                                       filter_no_act=True)
    aug = dict(protein_aug=protein_aug, molecule_aug=molecule_aug, text_aug=text_aug)
    train_dataset = lines_to_dataset(train_lines, node_index_manager, batch_size, shuffle=True, **aug)
    val_dataset = lines_to_dataset(val_lines, node_index_manager, batch_size, shuffle=False, **aug)
    test_dataset = lines_to_dataset(test_lines, node_index_manager, batch_size, shuffle=False, **aug)
    print(len(train_lines), len(train_dataset))

    model = MultiModalSeq(emb_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1 / (protein_aug + text_aug + molecule_aug)]).to(device))
    save_dir, score_file = prepare_files(f'seq_{args.name}')

    for epoch in range(epochs):
        run_epoch(model, optimizer, loss_fn, train_dataset, "train", score_file)
        run_epoch(model, optimizer, loss_fn, val_dataset, "valid", score_file)
        run_epoch(model, optimizer, loss_fn, test_dataset, "test", score_file)
