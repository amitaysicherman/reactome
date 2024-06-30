import random
from collections import defaultdict
from itertools import combinations
from torch.utils.data import Sampler
from tqdm import tqdm

from common.data_types import Reaction
import numpy as np

from dataset.fuse_dataset import PairsDataset, SameNameBatchSampler
from dataset.index_manger import NodesIndexManager, get_from_args
import os
from common.data_types import EMBEDDING_DATA_TYPES, PRETRAINED_EMD, DNA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset_builder import get_reactions
from sklearn.metrics import roc_auc_score
from itertools import chain
from common.utils import prepare_files, TYPE_TO_VEC_DIM
from model.models import MultiModalLinearConfig, MiltyModalLinear, EmbModel
from torch.utils.data import Dataset
from common.path_manager import scores_path

EMBEDDING_DATA_TYPES = [x for x in EMBEDDING_DATA_TYPES if x != DNA]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def triples_from_reaction(reaction: Reaction, nodes_index_manager: NodesIndexManager, per_sample_count):
    elements = []

    reaction_elements = reaction.inputs + sum([x.entities for x in reaction.catalysis], [])  # + reaction.outputs

    for reaction_element in reaction_elements:
        node = nodes_index_manager.name_to_node[reaction_element.get_db_identifier()]
        elements.append(node.index)
    for act in [x.activity for x in reaction.catalysis]:
        node = nodes_index_manager.name_to_node["GO@" + act]
        elements.append(node.index)
    elements_set = {x for x in elements}
    elements = list(elements_set)

    triples = []  # anchor, positive, negative
    types = []
    for e1, e2 in combinations(elements, 2):
        for i in range(per_sample_count):
            type_1 = nodes_index_manager.index_to_node[e1].type
            type_2 = nodes_index_manager.index_to_node[e2].type
            while True:
                fake_e1 = random.choice(nodes_index_manager.type_to_indexes[type_1])
                if fake_e1 not in elements_set:
                    break
            while True:
                fake_e2 = random.choice(nodes_index_manager.type_to_indexes[type_2])
                if fake_e2 not in elements_set:
                    break
            triples.append((e1, e2, fake_e2))
            types.append((type_1, type_2))

            triples.append((e2, e1, fake_e1))
            types.append((type_2, type_1))
    return types, triples


class TriplesDataset:
    def __init__(self, reactions, nodes_index_manager: NodesIndexManager, per_sample_count, batch_size, shuffle=False):
        self.nodes_index_manager = nodes_index_manager
        self.data = defaultdict(list)
        for reaction in tqdm(reactions):
            types, triples = triples_from_reaction(reaction, nodes_index_manager, per_sample_count=per_sample_count)
            for i in range(len(triples)):
                self.data[types[i]].append(triples[i])
        self.batch_size = batch_size

        self.batch_data = []
        for types, indexes in self.data.items():
            for i in range(0, len(indexes) - self.batch_size, self.batch_size):
                self.batch_data.append((types, indexes[i:i + self.batch_size]))
            if len(indexes) % self.batch_size != 0:
                self.batch_data.append((types, indexes[-(len(indexes) % self.batch_size):]))
        if shuffle:
            random.shuffle(self.batch_data)

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, index):
        (type1, type2), indexes = self.batch_data[index]
        anchors = np.stack([self.nodes_index_manager.index_to_node[i[0]].vec for i in indexes])
        positives = np.stack([self.nodes_index_manager.index_to_node[i[1]].vec for i in indexes])
        negatives = np.stack([self.nodes_index_manager.index_to_node[i[2]].vec for i in indexes])
        return (type1, type2), (anchors, positives, negatives)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def run_epoch(model, optimizer, loader, loss_func: nn.TripletMarginWithDistanceLoss, output_file, part, all_to_one,
              self_move):
    is_train = part == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    for (type1, type2), (anchors, positives, negatives) in loader:
        anchors = torch.from_numpy(anchors).to(device).float()
        positives = torch.from_numpy(positives).to(device).float()
        negatives = torch.from_numpy(negatives).to(device).float()
        all_labels.extend([1] * len(anchors) + [0] * len(anchors))

        if all_to_one != "":
            if (not self_move) and all_to_one == type1 and type1 == type2:
                all_preds.extend((0.5 * (1 + F.cosine_similarity(anchors, positives).cpu().detach().numpy())).tolist())
                all_preds.extend((0.5 * (1 + F.cosine_similarity(anchors, negatives).cpu().detach().numpy())).tolist())
                continue

            if all_to_one == type1:
                out1 = anchors.detach()
                out2 = model(positives, type2)
                out3 = model(negatives, type2)
            elif all_to_one == type2:
                out1 = model(anchors, type1)
                out2 = positives.detach()
                out3 = negatives.detach()
            else:
                out1 = model(anchors, type1)
                out2 = model(positives, type2)
                out3 = model(negatives, type2)
                all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist())
                all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out3).cpu().detach().numpy())).tolist())
                continue
        else:
            out1 = model(anchors, type1)
            out2 = model(positives, type2)
            out3 = model(negatives, type2)

        all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist())
        all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out3).cpu().detach().numpy())).tolist())

        loss = loss_func(out1, out2, out3)
        total_loss += loss.item()

        if not is_train:
            continue

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    auc = roc_auc_score(all_labels, all_preds)
    msg = f"{part} AUC {auc:.3f} LOSS {total_loss / len(loader):.3f}"
    with open(output_file, "a") as f:
        f.write(msg + "\n")
    print(msg)
    return auc


def build_models(fuse_output_dim, fuse_n_layers, fuse_hidden_dim, fuse_dropout, save_dir):
    model_config = MultiModalLinearConfig(
        embedding_dim=[TYPE_TO_VEC_DIM[x] for x in EMBEDDING_DATA_TYPES],
        n_layers=fuse_n_layers,
        names=EMBEDDING_DATA_TYPES,
        hidden_dim=fuse_hidden_dim,
        output_dim=[fuse_output_dim] * len(EMBEDDING_DATA_TYPES),
        dropout=fuse_dropout,
        normalize_last=1
    )
    model = MiltyModalLinear(model_config).to(device)
    model_config.save_to_file(f"{save_dir}/config.txt")
    return model


if __name__ == '__main__':

    from common.args_manager import get_args

    args = get_args()
    save_dir, scores_file = prepare_files(f'fuse_trip_{args.fuse_name}', skip_if_exists=args.skip_if_exists)
    node_index_manager = NodesIndexManager(pretrained_method=PRETRAINED_EMD, fuse_name="no")
    train_reactions, validation_reactions, test_reaction = get_reactions(filter_untrain=False,
                                                                         filter_dna=True,
                                                                         filter_no_act=True,
                                                                         sample_count=args.gnn_sample)
    ds_args = {"nodes_index_manager": node_index_manager, "batch_size": args.fuse_batch_size, "per_sample_count": 3}
    train_loader = TriplesDataset(train_reactions, **ds_args, shuffle=True)
    valid_loader = TriplesDataset(validation_reactions, **ds_args, shuffle=False)
    test_loader = TriplesDataset(test_reaction, **ds_args, shuffle=False)

    model = build_models(args.fuse_output_dim, args.fuse_n_layers, args.fuse_hidden_dim, args.fuse_dropout, save_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.fuse_lr)
    loss_func = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    running_args = {"model": model, "optimizer": optimizer, "loss_func": loss_func, "output_file": scores_file,
                    "all_to_one": args.fuse_all_to_one, "self_move": args.fuse_self_move}

    best_valid_auc = 0
    best_test_auc = 0
    best_index = 0
    for epoch in range(args.fuse_epochs):
        train_auc = run_epoch(**running_args, loader=train_loader, part="train")
        with torch.no_grad():
            valid_auc = run_epoch(**running_args, loader=valid_loader, part="valid")
            test_auc = run_epoch(**running_args, loader=test_loader, part="test")

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            best_index = epoch
            torch.save(model.state_dict(), f"{save_dir}/fuse_trip_model.pt")
    with open(os.path.join(scores_path, "fuse_trip_all.csv"), "a") as f:
        f.write(f"{best_valid_auc:.3f},{best_test_auc:.3f},{best_index}\n")
