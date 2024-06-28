import random
from typing import List
from collections import defaultdict
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
from dataset.seq_dataset import reaction_to_nodes
import os
from typing import Dict
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)


class MultiModalSeqHidden(nn.Module):
    def __init__(self, type_to_vec_dim: Dict[str, int], output_dim=1):
        super(MultiModalSeqHidden, self).__init__()
        self.d_types = [PROTEIN, MOLECULE, TEXT]

        self.emb_dim = 64
        self.t = nn.ModuleDict({k: nn.Linear(type_to_vec_dim[k], self.emb_dim) for k in self.d_types})
        self.trans1 = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=2, dim_feedforward=self.emb_dim * 2,
                                                 batch_first=True)
        self.trans2 = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=2, dim_feedforward=self.emb_dim * 2,
                                                 batch_first=True)
        self.last_lin = nn.Linear(self.emb_dim, output_dim)

    def forward(self, batch_data: Dict[str, torch.Tensor], batch_mask: Dict[str, torch.Tensor], return_prot_emd=False):
        all_transformed_data = []
        all_masks = []
        hidden_states = []
        for dtype in self.d_types:
            transformed_data = self.t[dtype](batch_data[dtype])
            mask = batch_mask[dtype].unsqueeze(-1)
            all_transformed_data.append(transformed_data)
            all_masks.append(mask)
        concatenated_data = torch.cat(all_transformed_data, dim=1)
        concatenated_mask = torch.cat(all_masks, dim=1)
        src_key_padding_mask = concatenated_mask.squeeze(-1) == 0
        hidden_states.append(concatenated_data)
        concatenated_data = self.trans1(concatenated_data, src_key_padding_mask=src_key_padding_mask)
        hidden_states.append(concatenated_data)
        concatenated_data = self.trans2(concatenated_data, src_key_padding_mask=src_key_padding_mask)
        hidden_states.append(concatenated_data)

        masked_data = concatenated_data * concatenated_mask
        sum_masked_data = masked_data.sum(dim=1)
        count_masked_data = concatenated_mask.sum(dim=1)

        mean_masked_data = sum_masked_data / torch.clamp(count_masked_data, min=1.0)

        output = self.last_lin(mean_masked_data)
        return output, hidden_states, concatenated_mask

    def get_emb_size(self):
        return self.emb_dim


def get_empty_dict():
    return {PROTEIN: [], MOLECULE: [], TEXT: []}


class ReactionSeqInd:
    def __init__(self, nodes: List[NodeData]):
        self.nodes = sorted(nodes, key=lambda x: x.type)
        self.replace_indexes = []

    def get_augment_copy(self, node_index_manager: NodesIndexManager):
        # TODO : if change to more then one replace index, need to change other parts of the code
        fake_index = random.choice(range(len(self.nodes)))
        entity_index = self.nodes[fake_index].index
        new_index = node_index_manager.sample_entity(entity_index, "random", self.nodes[fake_index].type)
        copy_data = ReactionSeqInd(self.nodes)
        copy_data.nodes[fake_index] = node_index_manager.index_to_node[new_index]
        copy_data.replace_indexes = [fake_index]
        return copy_data

    def to_vecs_types(self):
        type_to_vecs = get_empty_dict()
        for node in self.nodes:
            type_to_vecs[node.type].append(torch.Tensor(node.vec).float())
        return [(k, v) for k, v in type_to_vecs.items()]


def collate_fn(reaction_seqs: List[ReactionSeqInd], type_to_vec_dim):
    data_by_type = defaultdict(list)
    labels = []
    replace_indexes = []
    for reaction_seq in reaction_seqs:
        labels.append(len(reaction_seq.replace_indexes) > 0)
        replace_indexes.append(reaction_seq.replace_indexes)
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
    return batch_data, batch_mask, torch.tensor(labels), replace_indexes


def lines_to_dataset(lines, node_index_manager: NodesIndexManager, batch_size, shuffle, aug_factor, type_to_vec_dim):
    dataset = []
    for reaction in tqdm(lines):
        nodes = reaction_to_nodes(reaction, node_index_manager)
        data = ReactionSeqInd(nodes)
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


def hidden_states_to_pairs(emb, mask, replace_indexes, k=4):
    input1 = []
    input2 = []
    target = []
    # selected_samples = []
    mask = (mask == 1).squeeze(-1)

    for index_in_batch, replace_index in enumerate(replace_indexes):

        indexes_in_row = random.choices(range(mask[index_in_batch].sum().item()), k=k)
        if len(replace_index) > 1:
            replace_index = replace_index[0]  # TODO handle multiple replace indexes
            if replace_index not in indexes_in_row:
                indexes_in_row[0] = replace_index
        emb_index = emb[index_in_batch, mask[index_in_batch]][indexes_in_row]
        for i, j in itertools.combinations(range(k), 2):
            if i == j:
                continue
            input1.append(emb_index[i])
            input2.append(emb_index[j])
            target.append(1 if i == replace_index or j == replace_index else 0)  # TODO handle multiple replace indexes

    input1 = torch.stack(input1)
    input2 = torch.stack(input2)
    target = torch.tensor(target).to(device)
    return nn.functional.cosine_embedding_loss(input1, input2, target)


def run_epoch(model, optimizer, loss_fn, dataset, part, output_file, k, alphas, hiddens_bool):
    is_train = part == "train"
    y_real = []
    y_pred = []
    if is_train:
        model.train()
    else:
        model.eval()
    for batch_data, batch_mask, labels, replace_indexes in dataset:
        optimizer.zero_grad()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        batch_mask = {k: v.to(device) for k, v in batch_mask.items()}
        output, hidden_states, concatenated_mask = model(batch_data, batch_mask)
        y_real.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(torch.sigmoid(output).detach().cpu().numpy().tolist())

        pair_loss = 0
        hidden_states = [hidden_states[i] for i in range(len(hidden_states)) if hiddens_bool[i]]
        for alpha, hidden in zip(alphas, hidden_states):
            pair_loss += alpha * hidden_states_to_pairs(hidden, concatenated_mask, replace_indexes, k=k)
        loss = loss_fn(output, labels.float().unsqueeze(-1).to(device))
        alpha_total = sum(alphas)
        total_loss = (1 - alpha_total) * loss + pair_loss
        if is_train:
            total_loss.backward()
            optimizer.step()
    all_auc = roc_auc_score(y_real, y_pred)
    print(f"{part} AUC: {all_auc * 100:.2f}")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{part} AUC: {all_auc * 100:.2f}\n")
    return all_auc


if __name__ == "__main__":
    from common.args_manager import get_args

    args = get_args()

    batch_size = 128
    lr = 0.001
    aug_factor = args.seq_aug_factor
    alphas = args.seq_a
    hiddens_bool = args.seq_hidden
    assert len(hiddens_bool) == 3
    assert sum(hiddens_bool) == len(alphas)
    k = args.seq_k
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

    model = MultiModalSeqHidden(TYPE_TO_VEC_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1 / aug_factor]).to(device))
    save_dir, score_file = prepare_files(f'seq_const_{args.name}')

    best_score = 0
    best_prev_index = -1
    share_epoch_args = {"model": model, "optimizer": optimizer, "loss_fn": loss_fn, "k": k, "alphas": alphas,
                        "hiddens_bool": hiddens_bool, "output_file": score_file}
    for epoch in range(args.gnn_epochs):
        print(epoch)
        run_epoch(**share_epoch_args, dataset=train_dataset, part="train")
        with torch.no_grad():
            score = run_epoch(**share_epoch_args, dataset=val_dataset, part="val")
            run_epoch(**share_epoch_args, dataset=test_dataset, part="test")
