import numpy as np
import pandas as pd

from index_manger import NodesIndexManager, NodeTypes, EMBEDDING_DATA_TYPES
from sklearn.metrics.pairwise import cosine_distances

from biopax_parser import reaction_from_str, Reaction
from itertools import combinations
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset_builder import have_unkown_nodes
from collections import defaultdict
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
# try:
# from cuml import TSNE
# except:
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set()

root = "data/items"

TEST_MODE = True


def pairs_from_reaction(reaction: Reaction, nodes_index_manager: NodesIndexManager):
    elements = []
    reaction_elements = reaction.inputs + reaction.outputs + sum([x.entities for x in reaction.catalysis], [])
    for reaction_element in reaction_elements:
        node = nodes_index_manager.name_to_node[reaction_element.get_unique_id()]
        elements.append(node.index)
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


class TranferModel(nn.Module):
    def __init__(self, embedding_dim, n_layers=3, hidden_dim=256, output_dim=128):
        super(TranferModel, self).__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.ModuleDict({k: nn.Linear(v, output_dim) for k, v in embedding_dim.items()}))
        else:
            self.layers.append(nn.ModuleDict({k: nn.Linear(v, hidden_dim) for k, v in embedding_dim.items()}))
            for _ in range(n_layers - 2):
                self.layers.append(
                    nn.ModuleDict({k: nn.Linear(hidden_dim, hidden_dim) for k, v in embedding_dim.items()}))
            self.layers.append(nn.ModuleDict({k: nn.Linear(hidden_dim, output_dim) for k, v in embedding_dim.items()}))

    def forward(self, x, type_):
        x = F.normalize(x, dim=-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer[type_](x))
        x = self.layers[-1][type_](x)
        return F.normalize(x, dim=-1)


def get_two_pairs_without_share_nodes(node_index_manager: NodesIndexManager):
    a_elements = []
    b_elements = []
    for dtype in EMBEDDING_DATA_TYPES:
        a_elements.append(node_index_manager.dtype_to_first_index[dtype])
        a_elements.append(node_index_manager.dtype_to_first_index[dtype] + 1)
        b_elements.append(node_index_manager.dtype_to_first_index[dtype] + 2)
        b_elements.append(node_index_manager.dtype_to_first_index[dtype] + 3)
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
    def __init__(self, root, nodes_index_manager: NodesIndexManager, neg_count=1, test_mode=TEST_MODE, split="train"):
        self.nodes_index_manager = nodes_index_manager
        if test_mode:
            self.data = get_two_pairs_without_share_nodes(nodes_index_manager)
            self.elements_unique = np.array(list(set([x[0] for x in self.data] + [x[1] for x in self.data])))
            return
        with open(f'{root}/reaction.txt') as f:
            lines = f.readlines()
        lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
        reactions = [reaction_from_str(line) for line in lines]
        if split == "train":
            reactions = reactions[:int(len(reactions) * 0.8)]
        else:
            reactions = reactions[int(len(reactions) * 0.8):]
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
        """
        Initialize the sampler with the dataset and the batch size.
        """
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
        """
        Create an iterator that yields batches with the same name.
        """
        names = random.choices(self.names, k=len(self), weights=self.names_probs)
        for name in names:
            indices = self.name_to_indices[name]
            i = random.randint(0, len(indices) - self.batch_size)
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size


def indexes_to_tensor(indexes, node_index_manager: NodesIndexManager):
    type_ = node_index_manager.index_to_node[indexes[0].item()].type
    array = np.stack([node_index_manager.index_to_node[i.item()].vec for i in indexes])
    return torch.tensor(array), type_


def evel_model(pos_score, neg_score):
    pos_score_all_mean = np.average([np.mean(v) for k, v in pos_score.items()],
                                    weights=[len(v) for k, v in pos_score.items()])
    neg_score_all_mean = np.average([np.mean(v) for k, v in neg_score.items()],
                                    weights=[len(v) for k, v in neg_score.items()])
    pos_score = {k: np.mean(v) for k, v in pos_score.items()}
    neg_score = {k: np.mean(v) for k, v in neg_score.items()}

    pos_df = pd.DataFrame(columns=EMBEDDING_DATA_TYPES, index=EMBEDDING_DATA_TYPES)
    neg_df = pd.DataFrame(columns=EMBEDDING_DATA_TYPES, index=EMBEDDING_DATA_TYPES)
    for k in pos_score.keys():
        pos_df.loc[k[0], k[1]] = pos_score[k]
        neg_df.loc[k[0], k[1]] = neg_score[k]
    print("Positive scores")
    print(pos_score_all_mean)
    print(pos_df)
    print("Negative scores")
    print(neg_score_all_mean)
    print(neg_df)


def save_new_vecs(model, node_index_manager: NodesIndexManager):
    for dtype in EMBEDDING_DATA_TYPES:
        res = []
        for i in range(node_index_manager.dtype_to_first_index[dtype], node_index_manager.dtype_to_last_index[
            dtype]):
            node = node_index_manager.index_to_node[i]
            new_vec = model(torch.tensor(node.vec).to(device).unsqueeze(0), dtype).cpu().detach().numpy().flatten()
            if np.all(node.vec == 0):
                new_vec = np.zeros_like(new_vec)
            res.append(new_vec)

        res = np.stack(res)
        prev_v = np.load(f"{root}/{dtype}_vec.npy")

        assert len(res) == len(prev_v)

        np.save(f"{root}/{dtype}_vec_fuse.npy", res)


def evaluate_model(model, node_index_manager, element_to_show=None, n=100):
    # all_vecs = []
    # all_labels = []

    label_groups = {}
    for dtype in EMBEDDING_DATA_TYPES:
        nodes_in_type = [node for node in node_index_manager.index_to_node.values() if
                         node.type == dtype and np.any(node.vec != 0)]
        if element_to_show is not None:
            nodes_in_type = [node for node in nodes_in_type if node.index in element_to_show]
        vecs = np.stack([node.vec for node in nodes_in_type])
        vecs = model(torch.tensor(vecs).to(device), dtype).cpu().detach().numpy()
        # all_vecs.append(vecs)
        # all_labels.extend([dtype] * len(nodes_in_type))
        label_groups[dtype] = vecs
    res = pd.DataFrame(index=EMBEDDING_DATA_TYPES, columns=EMBEDDING_DATA_TYPES)
    for i, label_a in enumerate(EMBEDDING_DATA_TYPES):
        for label_b in EMBEDDING_DATA_TYPES[i:]:
            data_a = label_groups[label_a]
            data_b = label_groups[label_b]
            distances = np.mean(cosine_distances(data_a, data_b))
            res.loc[label_a, label_b] = distances
            res.loc[label_b, label_a] = distances
    print(res)
    # all_vecs = np.concatenate(all_vecs, axis=0)
    # all_labels = np.array(all_labels)
    # knn = NearestNeighbors(n_neighbors=n)
    #
    # knn.fit(all_vecs, all_labels)
    # for dtype in EMBEDDING_DATA_TYPES:
    #     neighbors = knn.kneighbors(all_vecs[all_labels == dtype], return_distance=False, n_neighbors=n)
    #     neighbors = neighbors.flatten()
    #     perp = all_labels[neighbors] != dtype
    #     print(f"{dtype}: {perp.mean()}", end="| ")
    # print()


def evel_test_split(model, node_index_manager, test_dataset):
    pos_score = {(i, j): [] for i in EMBEDDING_DATA_TYPES for j in EMBEDDING_DATA_TYPES}
    neg_score = {(i, j): [] for i in EMBEDDING_DATA_TYPES for j in EMBEDDING_DATA_TYPES}

    for idx1, idx2, label in test_dataset:
        data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
        data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)
        out1 = model(data_1.to(device), type_1)
        out2 = model(data_2.to(device), type_2)
        loss = contrastive_loss(out1, out2, label.to(device))
        if label.item() == 1:
            pos_score[(type_1, type_2)].append(loss.item())
        else:
            neg_score[(type_1, type_2)].append(loss.item())
    print("Test split")
    evel_model(pos_score, neg_score)


def visualize_data(model, node_index_manager, max_per_type=100, title="", element_to_show=None):
    all_vecs = []
    for dtype in EMBEDDING_DATA_TYPES:
        nodes_in_type = [node for node in node_index_manager.index_to_node.values() if
                         node.type == dtype and np.any(node.vec != 0)]
        if element_to_show is not None:
            nodes_in_type = [node for node in nodes_in_type if node.index in element_to_show]

        if max_per_type and len(nodes_in_type) > max_per_type:
            nodes_in_type = random.sample(nodes_in_type, max_per_type)
        vecs = np.stack([node.vec for node in nodes_in_type])
        vecs = model(torch.tensor(vecs).to(device), dtype).cpu().detach().numpy()
        all_vecs.append(vecs)
    perplexity = min(30, sum([len(x) for x in all_vecs]) // 5)
    vecs_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(
        np.concatenate(all_vecs, axis=0))
    prev_last_index = 0
    for i, dtype in enumerate(EMBEDDING_DATA_TYPES):
        vecs_2d_type = vecs_2d[prev_last_index:prev_last_index + len(all_vecs[i])]

        prev_last_index += len(all_vecs[i])
        s = 2 if not TEST_MODE else 25
        plt.scatter(vecs_2d_type[:, 0], vecs_2d_type[:, 1], label=dtype, s=s)
    plt.legend()
    if title:
        plt.title(title)
    # plt.savefig(f"data/fig/{title}.png", dpi=300)
    plt.show()


node_index_manager = NodesIndexManager(root)
dataset = PairsDataset(root, node_index_manager)
batch_size = 1024 if not TEST_MODE else 1
sampler = SameNameBatchSampler(dataset, batch_size)
loader = DataLoader(dataset, batch_sampler=sampler)

test_dataset = DataLoader(PairsDataset(root, node_index_manager, split="test"), 1)

emb_dim = {NodeTypes.protein: 1024, NodeTypes.molecule: 768, NodeTypes.dna: 768, NodeTypes.text: 768}
model = TranferModel(emb_dim).to(device)
# model.load_state_dict(torch.load("data/model/cont.pt")['model'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
contrastive_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
for epoch in range(100):
    evel_test_split(model, node_index_manager, test_dataset)
    total_loss = 0
    pos_score = {(i, j): [] for i in EMBEDDING_DATA_TYPES for j in EMBEDDING_DATA_TYPES}
    neg_score = {(i, j): [] for i in EMBEDDING_DATA_TYPES for j in EMBEDDING_DATA_TYPES}

    for i, (idx1, idx2, label) in enumerate(tqdm(loader)):
        data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
        data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)
        # if type_1 == type_2:
        #     continue

        optimizer.zero_grad()
        out1 = model(data_1.to(device), type_1)
        out2 = model(data_2.to(device), type_2)
        loss = contrastive_loss(out1, out2, label.to(device))

        pos_score[(type_1, type_2)].extend(loss[label == 1].cpu().detach().numpy().tolist())
        neg_score[(type_1, type_2)].extend(loss[label == -1].cpu().detach().numpy().tolist())

        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()

        optimizer.step()
    print(f"epoch {epoch} loss {total_loss / len(loader)}")
    total_loss = 0
    evel_model(pos_score, neg_score)
    save_new_vecs(model, node_index_manager)
