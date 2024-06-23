import random
from collections import defaultdict
from itertools import combinations

import numpy as np
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from common.data_types import Reaction
from dataset.dataset_builder import have_unkown_nodes, have_dna_nodes
from dataset.index_manger import NodesIndexManager
from common.data_types import EMBEDDING_DATA_TYPES

def pairs_from_reaction(reaction: Reaction, nodes_index_manager: NodesIndexManager):
    elements = []
    reaction_elements = reaction.inputs + reaction.outputs + sum([x.entities for x in reaction.catalysis], [])
    for reaction_element in reaction_elements:
        node = nodes_index_manager.name_to_node[reaction_element.get_db_identifier()]
        elements.append(node.index)
    for mod in sum([list(x.modifications) for x in reaction_elements], []):
        node = nodes_index_manager.name_to_node["TEXT@" + mod]
        elements.append(node.index)
    for act in [x.activity for x in reaction.catalysis]:
        node = nodes_index_manager.name_to_node["GO@" + act]
        elements.append(node.index)
    elements = [e for e in elements if nodes_index_manager.index_to_node[e].have_seq]
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
    def __init__(self, reactions, nodes_index_manager: NodesIndexManager, neg_count=1, test_mode=False, split="train"):
        self.nodes_index_manager = nodes_index_manager
        if test_mode:
            self.data = get_two_pairs_without_share_nodes(nodes_index_manager, split)
            self.elements_unique = np.array(list(set([x[0] for x in self.data] + [x[1] for x in self.data])))
            return

        reactions = [reaction for reaction in reactions if
                     not have_unkown_nodes(reaction, nodes_index_manager, check_output=True)]
        self.all_pairs = []
        self.all_elements = []
        for reaction in tqdm(reactions):
            elements, pairs = pairs_from_reaction(reaction, nodes_index_manager)
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
            a = random.choice(elements) if a_ is None else a_
            b = random.choice(elements) if b_ is None else b_
            # probs = getattr(self, f'{other_dtype}_probs')
            # a = random.choices(elements, weights=probs, k=1)[0] if a_ is None else a_
            # b = random.choices(elements, weights=probs, k=1)[0] if b_ is None else b_
            if (a, b) not in self.pairs_unique:
                return a, b

    def __getitem__(self, idx):
        return self.data[idx]


class SameNameBatchSampler(Sampler):
    def __init__(self, dataset: PairsDataset, batch_size, shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.name_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            idx1, idx2, _ = dataset[idx]
            type_1 = dataset.nodes_index_manager.index_to_node[idx1].type
            type_2 = dataset.nodes_index_manager.index_to_node[idx2].type
            self.name_to_indices[(type_1, type_2)].append(idx)
        if shuffle:
            for name in self.name_to_indices:
                random.shuffle(self.name_to_indices[name])
        self.names = list(self.name_to_indices.keys())
        if shuffle:
            random.shuffle(self.names)
        self.names_probs = np.array([len(indices) for indices in self.name_to_indices.values()])
        for i in range(len(self.names_probs)):
            print(self.names[i], self.names_probs[i])
        self.names_probs = self.names_probs / self.names_probs.sum()
        self.all_to_one = None

    def __iter__(self):
        for name in self.names:
            indices = self.name_to_indices[name]
            if self.batch_size > len(indices):
                yield indices
                continue
            for i in range(0, len(indices) - self.batch_size, self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size
