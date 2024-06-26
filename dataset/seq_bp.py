import random
import torch
import torch.nn as nn
from common.data_types import PROTEIN, TEXT, MOLECULE
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


def run_epoch(trans_model, model, optimizer, loss_fn, dataset, part, use_trans, output_file=""):
    is_train = part == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    real_labels = []
    pred_labels = []
    for batch_data, batch_mask, labels in dataset:
        optimizer.zero_grad()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        batch_mask = {k: v.to(device) for k, v in batch_mask.items()}
        emb = trans_model(batch_data, batch_mask, return_prot_emd=True)
        output = model(emb)
        labels = torch.tensor(labels)[
            sum([[i] * int(x.item()) for i, x in zip(range(len(labels)), batch_mask[PROTEIN].sum(dim=1))], [])]
        loss = loss_fn(output, labels.float().to(device))
        real_labels.append(labels.cpu().numpy())
        pred_labels.append(output.sigmoid().detach().cpu().numpy())

        print(loss)
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
    auc_sample = roc_auc_score(real_labels, pred_labels, average="samples")
    print(f"{part} AUC: {auc} AUC sample: {auc_sample}")
    return auc


if __name__ == "__main__":
    from common.args_manager import get_args

    batch_size = 2048
    lr = 0.01

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
    train_dataset = lines_to_dataset(train_lines, node_index_manager, bp_mapping, batch_size, shuffle=True,
                                     type_to_vec_dim=TYPE_TO_VEC_DIM)
    val_dataset = lines_to_dataset(val_lines, node_index_manager, bp_mapping, batch_size, shuffle=False,
                                   type_to_vec_dim=TYPE_TO_VEC_DIM)
    test_dataset = lines_to_dataset(test_lines, node_index_manager, bp_mapping, batch_size, shuffle=False,
                                    type_to_vec_dim=TYPE_TO_VEC_DIM)
    print(len(train_lines), len(train_dataset))

    trans_model = MultiModalSeq(args.seq_size, TYPE_TO_VEC_DIM, use_trans=args.seq_use_trans).to(device).eval()
    # if args.seq_use_trans:
    model = nn.Linear(trans_model.get_emb_size(), bp_mapping.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([bp_mapping.shape[0] / bp_mapping.sum().values]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    save_dir, score_file = prepare_files(f'bp_{args.name}')

    best_score = 0
    best_prev_index = -1
    epoch_args = dict(trans_model=trans_model, model=model, optimizer=optimizer, loss_fn=loss_fn,
                      output_file=score_file, use_trans=args.seq_use_trans)
    for epoch in range(args.gnn_epochs):
        print(f"Epoch {epoch}")
        run_epoch(**epoch_args, dataset=train_dataset, part="train")
        epoch_score = run_epoch(**epoch_args, dataset=val_dataset, part="valid")
        run_epoch(**epoch_args, dataset=test_dataset, part="test")

        if epoch_score > best_score:
            best_score = epoch_score
            torch.save(model.state_dict(), f"{save_dir}/{epoch}.pt")
            if best_prev_index != -1:
                os.remove(f"{save_dir}/{best_prev_index}.pt")
            best_prev_index = epoch
