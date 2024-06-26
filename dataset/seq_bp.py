import random
import torch
import torch.nn as nn
from common.data_types import PROTEIN, TEXT, MOLECULE, Reaction
from typing import List
from common.utils import prepare_files
from dataset.dataset_builder import get_reactions
from dataset.index_manger import NodesIndexManager, get_from_args
from model.models import MultiModalSeq
import os
from dataset.seq_dataset import reaction_to_nodes, ReactionSeq, collate_fn
import pandas as pd
from pybiopax.biopax import BiochemicalReaction
from pybiopax.biopax.base import Pathway
from pybiopax.biopax.util import RelationshipXref as RelXref
from pybiopax.biopax.interaction import Catalysis
import pybiopax
import numpy as np
from preprocessing.biopax_parser import get_reactome_id
from common.path_manager import data_path, item_path
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)


def reactions_to_biological_process():
    bp_file = f'{item_path}/biological_process.csv'
    if os.path.exists(bp_file):
        return pd.read_csv(bp_file, index_col=0)

    input_file = os.path.join(data_path, "biopax", "Homo_sapiens.owl")
    model = pybiopax.model_from_owl_file(input_file)
    pathways = list(model.get_objects_by_type(Pathway))
    inv_mapping = {}
    for pathway in pathways:
        for c in pathway.pathway_component:
            inv_mapping[c] = pathway
    reactions = list(model.get_objects_by_type(BiochemicalReaction))
    uid_to_id = {r.uid: get_reactome_id(r) for r in reactions}
    ids = [uid_to_id[r.uid] for r in reactions]
    biological_process = [ref.id for ref in model.get_objects_by_type(RelXref) if
                          ref.db == "GENE ONTOLOGY" and not isinstance(list(ref.xref_of)[0], Catalysis)]
    mapping = pd.DataFrame(index=ids, columns=biological_process)
    for reaction in reactions:
        id_ = uid_to_id[reaction.uid]
        while reaction:
            for go_id in [ref.id for ref in reaction.xref if isinstance(ref, RelXref) and ref.db == "GENE ONTOLOGY"]:
                mapping.loc[id_, go_id] = 1
            if reaction in inv_mapping:
                reaction = inv_mapping[reaction]
            else:
                reaction = None
    mapping = mapping.fillna(0)
    mapping.to_csv(bp_file)
    return mapping


def lines_to_dataset(lines, node_index_manager: NodesIndexManager, mapping, batch_size, shuffle, type_to_vec_dim):
    dataset = []

    for reaction in lines:
        nodes = reaction_to_nodes(reaction, node_index_manager)
        labels = mapping.loc[reaction.reactome_id].values
        data = ReactionSeq(nodes, labels)
        dataset.append(data)

    if shuffle:
        random.shuffle(dataset)
    all_data = []
    for i in range(0, len(dataset), batch_size):
        all_data.append(collate_fn(dataset[i:i + batch_size], return_bp=True, type_to_vec_dim=type_to_vec_dim))
    if len(dataset) % batch_size != 0:
        all_data.append(
            collate_fn(dataset[-(len(dataset) % batch_size):], return_bp=True, type_to_vec_dim=type_to_vec_dim))
    return all_data


def filter_part(mapping, lines):
    indexes = [r.reactome_id for r in lines]
    filtered_df = mapping.loc[indexes]
    filtered_indexes = set(list(filtered_df[filtered_df.sum(axis=1) != 0].index))
    print(f"Removed {len(indexes) - len(filtered_indexes)}/{len(indexes)} reactions")
    return [r for r in lines if r.reactome_id in filtered_indexes]


def filter_by_labels(bp_mapping, train_lines, val_lines, test_lines, min_per_label):
    train_lines = [x for x in train_lines if x.reactome_id in bp_mapping.index]
    val_lines = [x for x in val_lines if x.reactome_id in bp_mapping.index]
    test_lines = [x for x in test_lines if x.reactome_id in bp_mapping.index]

    column_sums = bp_mapping.loc[[x.reactome_id for x in train_lines]].sum()

    filtered_columns = column_sums[column_sums >= min_per_label].index
    print(f"Removed {len(bp_mapping.columns) - len(filtered_columns)}/{len(bp_mapping.columns)} labels")
    bp_mapping.drop(columns=[c for c in bp_mapping.columns if c not in filtered_columns], inplace=True)

    train_lines = filter_part(bp_mapping, train_lines)
    val_lines = filter_part(bp_mapping, val_lines)
    test_lines = filter_part(bp_mapping, test_lines)
    return train_lines, val_lines, test_lines, bp_mapping


def run_epoch(model, optimizer, loss_fn, X, y, part, output_file=""):
    is_train = part == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    real_labels = []
    pred_labels = []
    optimizer.zero_grad()

    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device)
    output = model(X)
    loss = loss_fn(output, y.float())
    print(loss.item())
    real_labels.append(y.cpu().numpy())
    pred_labels.append(output.sigmoid().detach().cpu().numpy())

    # print(loss)
    if is_train:
        loss.backward()
        optimizer.step()
    real_labels = np.concatenate(real_labels)
    pred_labels = np.concatenate(pred_labels)
    mask_zero = real_labels.sum(axis=0) != 0
    mask_one = real_labels.sum(axis=0) != len(real_labels)
    mask = mask_zero & mask_one
    real_labels = real_labels.T[mask].T
    pred_labels = pred_labels.T[mask].T
    if real_labels.shape[1] == 0:
        print(f"No labels for {part}")
        return 0
    auc = roc_auc_score(real_labels, pred_labels, average="weighted")
    # auc_sample = roc_auc_score(real_labels, pred_labels, average="samples")
    print(f"{part} Loss:{loss.item()} AUC: {auc}")  # AUC sample: {auc_sample}")
    with open(output_file, "a") as f:
        f.write(f"{part} Loss:{loss.item()} AUC: {auc}\n")
    return auc


def reaction_to_prots(reaction: Reaction, node_index_manager: NodesIndexManager):
    catalysis_enetities_nodes = [node_index_manager.name_to_node[e.get_db_identifier()] for c in reaction.catalysis for
                                 e in c.entities]
    input_nodes = [node_index_manager.name_to_node[e.get_db_identifier()] for e in reaction.inputs]
    reaction_nodes = catalysis_enetities_nodes + input_nodes
    nodes_id = [x.index for x in reaction_nodes if x.type == PROTEIN]
    return nodes_id


def get_labels_per_protein(node_index_manager: NodesIndexManager, reactions: List[Reaction], mapping: pd.DataFrame,
                           min_labels=3):
    protein_to_reactions = defaultdict(list)
    for reaction in reactions:
        for id_ in reaction_to_prots(reaction, node_index_manager):
            protein_to_reactions[id_].append(reaction.reactome_id)
    X = []
    Y = []
    for id_, reaction_list in protein_to_reactions.items():
        X.append(node_index_manager.index_to_node[id_].vec)
        Y.append(mapping.loc[reaction_list].sum(axis=0).values.astype(bool).astype(int))
    X = np.stack(X)
    Y = np.stack(Y)
    Y = Y[:, Y.sum(axis=0) > min_labels]
    mask = Y.sum(axis=1) > 0
    X = X[mask]
    Y = Y[mask]
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def trans_dataset_to_xy(dataset, trans_model):
    X = []
    Y = []
    for batch_data, batch_mask, labels in dataset:
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        batch_mask = {k: v.to(device) for k, v in batch_mask.items()}
        emb = trans_model(batch_data, batch_mask, return_prot_emd=True)
        emb = emb.reshape(-1, emb.shape[-1])
        protein_mask = batch_mask[PROTEIN]
        protein_mask = protein_mask.reshape(-1)
        emb = emb[protein_mask == 1]
        labels = torch.tensor(labels)[
            sum([[i] * int(x.item()) for i, x in zip(range(len(labels)), batch_mask[PROTEIN].sum(dim=1))], [])]
        labels = labels.cpu().numpy()
        emb = emb.detach().cpu().numpy()
        X.append(emb)
        Y.append(labels)
    return np.concatenate(X), np.concatenate(Y)


if __name__ == "__main__":
    from common.args_manager import get_args

    batch_size = 512
    lr = 0.1

    args = get_args()
    node_index_manager: NodesIndexManager = get_from_args(args)

    bp_mapping = reactions_to_biological_process()
    bp_mapping.index = bp_mapping.index.astype(int)
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
    train_lines, val_lines, test_lines, bp_mapping = filter_by_labels(bp_mapping, train_lines, val_lines, test_lines, 3)

    if args.seq_use_trans:
        trans_model = MultiModalSeq(args.seq_size, TYPE_TO_VEC_DIM, use_trans=args.seq_use_trans).to(device).eval()
        input_dim = trans_model.get_emb_size()
        train_dataset = lines_to_dataset(train_lines, node_index_manager, bp_mapping, batch_size, shuffle=True,
                                         type_to_vec_dim=TYPE_TO_VEC_DIM)
        val_dataset = lines_to_dataset(val_lines, node_index_manager, bp_mapping, batch_size, shuffle=False,
                                       type_to_vec_dim=TYPE_TO_VEC_DIM)
        test_dataset = lines_to_dataset(test_lines, node_index_manager, bp_mapping, batch_size, shuffle=False,
                                        type_to_vec_dim=TYPE_TO_VEC_DIM)

        X_train, y_train = trans_dataset_to_xy(train_dataset, trans_model)
        X_val, y_val = trans_dataset_to_xy(val_dataset, trans_model)
        X_test, y_test = trans_dataset_to_xy(test_dataset, trans_model)

    else:
        input_dim = TYPE_TO_VEC_DIM[PROTEIN]
        X_train, X_val, X_test, y_train, y_val, y_test = get_labels_per_protein(node_index_manager,
                                                                                train_lines + val_lines + test_lines,
                                                                                bp_mapping)
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    model = nn.Linear(input_dim, y_test.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # pos_weight = torch.tensor([bp_mapping.shape[0] / bp_mapping.sum().values]).to(device)
    pos_weight = y_train.shape[0] / y_train.sum()
    pos_weight = torch.tensor(pos_weight).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    save_dir, score_file = prepare_files(f'bp_{args.name}_{args.seq_use_trans}')
    best_score = 0
    best_prev_index = -1
    epoch_args = dict(model=model, optimizer=optimizer, loss_fn=loss_fn,
                      output_file=score_file)
    for epoch in range(args.gnn_epochs):
        print(f"Epoch {epoch}")
        run_epoch(**epoch_args, X=X_train, y=y_train, part="train")
        epoch_score = run_epoch(**epoch_args, X=X_val, y=y_val, part="valid")
        run_epoch(**epoch_args, X=X_test, y=y_test, part="test")

        # if epoch_score > best_score:
        #     best_score = epoch_score
        #     torch.save(model.state_dict(), f"{save_dir}/{epoch}.pt")
        #     if best_prev_index != -1:
        #         os.remove(f"{save_dir}/{best_prev_index}.pt")
        #     best_prev_index = epoch
