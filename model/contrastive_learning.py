import numpy as np
from dataset.index_manger import NodesIndexManager
import os
from common.utils import reaction_from_str
from common.data_types import Reaction, EMBEDDING_DATA_TYPES, NodeTypes
from itertools import combinations
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.dataset_builder import have_unkown_nodes, have_dna_nodes
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from sklearn.metrics import roc_auc_score
from itertools import chain
from common.path_manager import reactions_file, item_path, model_path, scores_path
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
TEST_MODE = False


def pairs_from_reaction(reaction: Reaction, nodes_index_manager: NodesIndexManager, proteins_molecules_only: bool):
    elements = []
    reaction_elements = reaction.inputs + reaction.outputs + sum([x.entities for x in reaction.catalysis], [])
    for reaction_element in reaction_elements:
        node = nodes_index_manager.name_to_node[reaction_element.get_db_identifier()]
        elements.append(node.index)
    if not proteins_molecules_only:
        for mod in sum([list(x.modifications) for x in reaction_elements], []):
            node = nodes_index_manager.name_to_node["TEXT@" + mod]
            elements.append(node.index)
        for act in [x.activity for x in reaction.catalysis]:
            node = nodes_index_manager.name_to_node["GO@" + act]
            elements.append(node.index)
    elements = [e for e in elements if not np.all(nodes_index_manager.index_to_node[e].vec == 0)]
    elements = list(set(elements))
    pairs = []
    for e1, e2 in combinations(elements, 2):
        type_1 = nodes_index_manager.index_to_node[e1].type
        type_2 = nodes_index_manager.index_to_node[e2].type

        if type_1 >= type_2:
            pairs.append((e2, e1))
        else:
            pairs.append((e1, e2))
    return elements, pairs


def get_two_pairs_without_share_nodes(node_index_manager: NodesIndexManager, split):
    a_elements = []
    b_elements = []
    split_factor = 0 if split == "train" else 4
    for dtype in EMBEDDING_DATA_TYPES:
        a_elements.append(node_index_manager.dtype_to_first_index[dtype] + split_factor)
        a_elements.append(node_index_manager.dtype_to_first_index[dtype] + 1 + split_factor)
        b_elements.append(node_index_manager.dtype_to_first_index[dtype] + 2 + split_factor)
        b_elements.append(node_index_manager.dtype_to_first_index[dtype] + 3 + split_factor)
    data = []
    for a1 in a_elements:
        for a2 in a_elements:
            if a1 != a2:
                data.append((a1, a2, 1))
    for a1 in a_elements:
        for b2 in b_elements:
            data.append((a1, b2, -1))
    for b1 in b_elements:
        for a2 in a_elements:
            data.append((b1, a2, -1))
    for b1 in b_elements:
        for b2 in b_elements:
            if b1 != b2:
                data.append((b1, b2, 1))
    return data


class PairsDataset(Dataset):
    def __init__(self, nodes_index_manager: NodesIndexManager, proteins_molecules_only: bool, neg_count=1,
                 test_mode=TEST_MODE, split="train"):
        self.nodes_index_manager = nodes_index_manager
        if test_mode:
            self.data = get_two_pairs_without_share_nodes(nodes_index_manager, split)
            self.elements_unique = np.array(list(set([x[0] for x in self.data] + [x[1] for x in self.data])))
            return
        with open(reactions_file) as f:
            lines = f.readlines()
        lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
        reactions = [reaction_from_str(line) for line in lines]
        if split == "train":
            reactions = reactions[:int(len(reactions) * 0.8)]
        else:
            reactions = reactions[int(len(reactions) * 0.8):]
        reactions = [reaction for reaction in reactions if
                     not have_unkown_nodes(reaction, nodes_index_manager, check_output=True)]
        if proteins_molecules_only:
            reactions = [reaction for reaction in reactions if
                         not have_dna_nodes(reaction, nodes_index_manager, check_output=True)]
        self.all_pairs = []
        self.all_elements = []
        for reaction in tqdm(reactions):
            elements, pairs = pairs_from_reaction(reaction, nodes_index_manager, proteins_molecules_only)
            self.all_elements.extend(elements)
            self.all_pairs.extend(pairs)
        self.pairs_unique = set(self.all_pairs)
        elements_unique, elements_count = np.unique(self.all_elements, return_counts=True)
        self.elements_unique = elements_unique
        for dtype in EMBEDDING_DATA_TYPES:
            dtype_indexes = [i for i in range(len(elements_unique)) if
                             nodes_index_manager.index_to_node[elements_unique[i]].type == dtype]
            dtype_unique = elements_unique[dtype_indexes]

            setattr(self, f"{dtype}_unique", dtype_unique)
            print(f"{dtype} unique: {len(dtype_unique)}")
        self.data = []
        for i in tqdm(range(len(self.all_pairs))):
            a, b = self.all_pairs[i]
            a_type = nodes_index_manager.index_to_node[a].type
            b_type = nodes_index_manager.index_to_node[b].type
            self.data.append((a, b, 1))
            self.data.append((b, a, 1))
            for i in range(neg_count):
                self.data.append((*self.sample_neg_pair(a_=a, other_dtype=b_type), -1))
                self.data.append((*self.sample_neg_pair(b_=b, other_dtype=a_type), -1))
                self.data.append((*self.sample_neg_pair(a_=b, other_dtype=a_type), -1))
                self.data.append((*self.sample_neg_pair(b_=a, other_dtype=b_type), -1))

    def __len__(self):
        return len(self.data)

    def sample_neg_pair(self, a_=None, b_=None, other_dtype=None):
        while True:
            elements = getattr(self, f'{other_dtype}_unique')
            # probs = getattr(self, f'{other_dtype}_probs')
            a = random.choice(elements) if a_ is None else a_
            b = random.choice(elements) if b_ is None else b_
            # a = random.choices(elements, weights=probs, k=1)[0] if a_ is None else a_
            # b = random.choices(elements, weights=probs, k=1)[0] if b_ is None else b_
            if (a, b) not in self.pairs_unique:
                return a, b

    def __getitem__(self, idx):
        return self.data[idx]


class SameNameBatchSampler(Sampler):
    def __init__(self, dataset: PairsDataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.name_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            idx1, idx2, _ = dataset[idx]
            type_1 = dataset.nodes_index_manager.index_to_node[idx1].type
            type_2 = dataset.nodes_index_manager.index_to_node[idx2].type
            self.name_to_indices[(type_1, type_2)].append(idx)
        for name in self.name_to_indices:
            random.shuffle(self.name_to_indices[name])
        self.names = list(self.name_to_indices.keys())
        self.names_probs = np.array([len(indices) for indices in self.name_to_indices.values()])
        for i in range(len(self.names_probs)):
            print(self.names[i], self.names_probs[i])
        self.names_probs = self.names_probs / self.names_probs.sum()

    def __iter__(self):
        names = random.choices(self.names, k=len(self), weights=self.names_probs)
        names += self.names
        for name in names:
            indices = self.name_to_indices[name]
            if self.batch_size > len(indices):
                yield indices
                continue
            i = random.randint(0, len(indices) - self.batch_size)
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size


def indexes_to_tensor(indexes, node_index_manager: NodesIndexManager):
    type_ = node_index_manager.index_to_node[indexes[0].item()].type
    array = np.stack([node_index_manager.index_to_node[i.item()].vec for i in indexes])
    return torch.tensor(array), type_


def remove_vecs_files(run_name):
    if not os.path.exists(f"{item_path}/{run_name}"):
        return
    for file_name in os.listdir(f"{item_path}/{run_name}"):
        if file_name.endswith(".npy"):
            os.remove(f"{item_path}/{run_name}/{file_name}")


def save_fuse_model(model: MiltyModalLinear, reconstruction_model: MiltyModalLinear, save_dir, epoch):
    output_file = f"{save_dir}/fuse_{epoch}.pt"
    torch.save(model.state_dict(), output_file)
    output_file = f"{save_dir}/fuse-recon_{epoch}.pt"
    torch.save(reconstruction_model.state_dict(), output_file)


def weighted_mean_loss(loss, labels):
    positive_mask = (labels == 1).float().to(device)
    negative_mask = (labels == -1).float().to(device)

    pos_weight = 1.0 / (max(positive_mask.sum(), 1))
    neg_weight = 1.0 / (max(negative_mask.sum(), 1))

    positive_loss = (loss * positive_mask).sum() * pos_weight
    negative_loss = (loss * negative_mask).sum() * neg_weight

    return (positive_loss + negative_loss) / (pos_weight + neg_weight)


def run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, loader, contrastive_loss, epoch, recon,
              output_file, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_recon_loss = 0
    all_labels = []
    all_preds = []

    for i, (idx1, idx2, label) in enumerate(loader):
        data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
        data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)
        data_1 = data_1.to(device).float()
        data_2 = data_2.to(device).float()
        out1 = model(data_1, type_1)
        out2 = model(data_2, type_2)
        all_labels.extend((label == 1).cpu().detach().numpy().astype(int).tolist())
        all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist())
        cont_loss = contrastive_loss(out1, out2, label.to(device))
        total_loss += cont_loss.mean().item()
        recon_1 = reconstruction_model(out1, type_1)
        recon_2 = reconstruction_model(out2, type_2)
        recon_loss = F.mse_loss(recon_1, data_1) + F.mse_loss(recon_2, data_2)
        total_recon_loss += recon_loss.item()
        if not is_train:
            continue
        if not recon or i % 2 == 0:
            cont_loss = weighted_mean_loss(cont_loss, label)
            cont_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            recon_loss.backward()
            reconstruction_optimizer.step()
            reconstruction_optimizer.zero_grad()
    auc = roc_auc_score(all_labels, all_preds)

    msg = f"Epoch {epoch} {'Train' if is_train else 'Test'} AUC {auc:.3f} (cont: {total_loss / len(loader):.3f}, " \
          f"recon: {total_recon_loss / len(loader):.3f})"
    with open(output_file, "a") as f:
        f.write(msg + "\n")
    print(msg)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--proteins_molecules_only", type=int, default=0)
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--recon", type=int, default=1)
    parser.add_argument("--name", type=str, default="all-recon")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    EPOCHS = args.epochs

    run_name = args.name
    if not TEST_MODE:
        remove_vecs_files(run_name)

    if args.proteins_molecules_only:
        EMBEDDING_DATA_TYPES = [NodeTypes.protein, NodeTypes.molecule]
    if TEST_MODE:
        args.batch_size = 2
    node_index_manager = NodesIndexManager()
    dataset = PairsDataset(node_index_manager, proteins_molecules_only=args.proteins_molecules_only)

    sampler = SameNameBatchSampler(dataset, args.batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler)

    test_dataset = PairsDataset(node_index_manager, split="test",
                                proteins_molecules_only=args.proteins_molecules_only)
    test_sampler = SameNameBatchSampler(test_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    if args.proteins_molecules_only:
        emb_dim = {NodeTypes.protein: 1024, NodeTypes.molecule: 768}
    else:
        emb_dim = {NodeTypes.protein: 1024, NodeTypes.molecule: 768, NodeTypes.dna: 768, NodeTypes.text: 768}

    save_dir = f"{model_path}/fuse_{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file_name in os.listdir(save_dir):
        if file_name.endswith(".pt"):
            os.remove(f"{save_dir}/{file_name}")
    scores_file = f"{scores_path}/fuse_{run_name}.txt"
    if os.path.exists(scores_file):
        os.remove(scores_file)
    model_config = MultiModalLinearConfig(
        embedding_dim=list(emb_dim.values()),
        n_layers=args.n_layers,
        names=list(emb_dim.keys()),
        hidden_dim=args.hidden_dim,
        output_dim=[args.output_dim] * len(emb_dim),
        dropout=args.dropout,
        normalize_last=1
    )

    model = MiltyModalLinear(model_config).to(device)

    model_config.save_to_file(f"{save_dir}/config.txt")

    recons_config = MultiModalLinearConfig(
        embedding_dim=[args.output_dim] * len(emb_dim),
        n_layers=args.n_layers,
        names=list(emb_dim.keys()),
        hidden_dim=args.hidden_dim,
        output_dim=list(emb_dim.values()),
        dropout=args.dropout,
        normalize_last=0
    )
    recons_config.save_to_file(f"{save_dir}/config-recon.txt")

    reconstruction_model = MiltyModalLinear(recons_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reconstruction_optimizer = torch.optim.Adam(chain(model.parameters(), reconstruction_model.parameters()),
                                                lr=args.lr)
    contrastive_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    for epoch in range(EPOCHS):
        run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, test_loader, contrastive_loss,
                  epoch, args.recon, scores_file, is_train=False)

        run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, loader, contrastive_loss, epoch,
                  args.recon, scores_file, is_train=True)

        save_fuse_model(model, reconstruction_model, save_dir, epoch)
