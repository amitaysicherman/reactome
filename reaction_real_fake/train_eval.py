import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from common.data_types import REAL, PROTEIN, TEXT, MOLECULE
from common.data_types import Reaction
from common.utils import prepare_files
from dataset.dataset_builder import get_reactions, add_if_not_none
from dataset.index_manger import NodesIndexManager, NodeData, get_from_args
from model.models import MultiModalSeq
from common.path_manager import scores_path
import os
from torchdrug.metrics import area_under_roc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)


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
    real = []
    pred = []
    for batch_data, batch_mask, labels, augmentations in dataset:
        optimizer.zero_grad()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        batch_mask = {k: v.to(device) for k, v in batch_mask.items()}
        output = model(batch_data, batch_mask)
        labels = labels.float().unsqueeze(-1).to(device)
        loss = loss_fn(output, labels)
        output = torch.sigmoid(output).detach()
        real.append(labels)
        pred.append(output)
        if is_train:
            loss.backward()
            optimizer.step()
    if part != "train":
        real = torch.cat(real, dim=0)
        pred = torch.cat(pred, dim=0)
        score = area_under_roc(pred.flatten(), real.flatten()).item()
        return score
    else:
        return 0


def main(args, model=None):
    batch_size = 2048
    lr = 0.001
    aug_factor = 5

    max_no_improve = args.max_no_improve
    node_index_manager = NodesIndexManager(pretrained_method=args.gnn_pretrained_method, fuse_name=args.fuse_name,
                                           fuse_pretrained_start=args.fuse_pretrained_start,
                                           prot_emd_type=args.protein_emd,
                                           mol_emd_type=args.mol_emd, fuse_model=model)
    TYPE_TO_VEC_DIM = {PROTEIN: node_index_manager.index_to_node[node_index_manager.protein_indexes[0]].vec.shape[0],
                       MOLECULE: node_index_manager.index_to_node[node_index_manager.molecule_indexes[0]].vec.shape[0],
                       TEXT: node_index_manager.index_to_node[node_index_manager.text_indexes[0]].vec.shape[0]}

    train_lines, val_lines, test_lines = get_reactions(args.gnn_sample, filter_dna=True, filter_no_act=True)
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

    best_val_score = 0
    best_test_score = 0
    n_no_improve = 0
    for epoch in range(args.gnn_epochs):
        run_epoch(model, optimizer, loss_fn, train_dataset, "train", score_file)
        with torch.no_grad():
            val_score = run_epoch(model, optimizer, loss_fn, val_dataset, "valid", score_file)
            test_score = run_epoch(model, optimizer, loss_fn, test_dataset, "test", score_file)

        if args.dp_print:
            print(epoch, val_score, test_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_test_score = test_score
            n_no_improve = 0
        else:
            n_no_improve += 1
            if n_no_improve == max_no_improve:
                break
    if args.dp_print:
        print("Best Test scores\n", best_test_score)
        output_file = f"{scores_path}/rrf.csv"
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write(f'{args.name},{args.gnn_pretrained_method},{args.protein_emd},{args.mol_emd},auc\n')
        with open(output_file, "a") as f:
            f.write(f'{args.name},{args.gnn_pretrained_method},{args.protein_emd},{args.mol_emd},{best_test_score}\n')
    return best_val_score, best_test_score


if __name__ == "__main__":
    from common.args_manager import get_args

    args = get_args()
    main(args)
