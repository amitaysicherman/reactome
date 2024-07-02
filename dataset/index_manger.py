import numpy as np
from sklearn.metrics import pairwise_distances
import random
from typing import Dict
from common.utils import TYPE_TO_VEC_DIM, load_fuse_model
from common.path_manager import item_path
from common.data_types import REACTION, COMPLEX, UNKNOWN_ENTITY_TYPE, PROTEIN, EMBEDDING_DATA_TYPES, LOCATION, \
    DATA_TYPES, NodeTypes, BIOLOGICAL_PROCESS, NO_PRETRAINED_EMD, PRETRAINED_EMD, PRETRAINED_EMD_FUSE,MOLECULE,TEXT
from model.models import apply_model
from functools import lru_cache
import os
import torch

REACTION_NODE_ID = 0
COMPLEX_NODE_ID = 1
UNKNOWN_ENTITY_TYPE = UNKNOWN_ENTITY_TYPE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(42)
class NodeData:
    def __init__(self, index, name, type_, vec=None, have_seq=True):
        self.index = index
        self.name = name
        self.type = type_
        self.vec = vec
        self.have_seq = have_seq


class NodesIndexManager:
    def __init__(self, pretrained_method=PRETRAINED_EMD, fuse_name="", fuse_pretrained_start=True):
        reaction_node = NodeData(REACTION_NODE_ID, REACTION, NodeTypes.reaction)
        complex_node = NodeData(COMPLEX_NODE_ID, COMPLEX, NodeTypes.complex)
        self.nodes = [reaction_node, complex_node]
        locations_counts = {}
        self.index_count = 2
        self.dtype_to_first_index = dict()
        self.dtype_to_last_index = dict()
        self.bp_name_to_index = {"": -1}
        self.fuse_model = load_fuse_model(fuse_name, fuse_pretrained_start)
        if self.fuse_model is not None:
            self.fuse_model.to(device)
        with open(f'{item_path}/{BIOLOGICAL_PROCESS}.txt') as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            bp = line.split("@")[0]
            self.bp_name_to_index[bp] = i if bp != "" else -1
        for dt in DATA_TYPES:
            self.dtype_to_first_index[dt] = self.index_count
            names_file = f'{item_path}/{dt}.txt'
            with open(names_file) as f:
                lines = f.read().splitlines()

            if dt in EMBEDDING_DATA_TYPES:
                if pretrained_method == NO_PRETRAINED_EMD:
                    vectors = np.stack([np.random.rand(TYPE_TO_VEC_DIM[dt]) for _ in range(len(lines))])
                else:
                    pretrained_vec_file = f'{item_path}/{dt}_vec.npy'
                    vectors = np.load(pretrained_vec_file)
                    if pretrained_method == PRETRAINED_EMD_FUSE:
                        if fuse_pretrained_start:
                            with torch.no_grad():
                                vectors = apply_model(self.fuse_model, vectors, dt).detach().cpu().numpy()
                        else:
                            vectors = self.fuse_model.emd.weight.detach().cpu().numpy()[
                                      self.index_count:self.index_count + len(lines)]
            elif dt == UNKNOWN_ENTITY_TYPE:
                vectors = [np.zeros(TYPE_TO_VEC_DIM[PROTEIN]) for _ in range(len(lines))]
            else:
                vectors = [None] * len(lines)

            seq_file = f'{item_path}/{dt}_sequences.txt'
            if not os.path.exists(seq_file):
                seqs = [False] * len(lines)
            else:
                with open(seq_file) as f:
                    seqs = f.read().splitlines()
                    seqs = [True if len(seq) > 0 else False for seq in seqs]

            for i, line in enumerate(lines):
                name = "@".join(line.split("@")[:-1])
                node = NodeData(self.index_count, name, dt, vectors[i], seqs[i])
                self.nodes.append(node)
                self.index_count += 1
                if dt == LOCATION:
                    count = line.split("@")[-1]
                    locations_counts[node.index] = int(count)
            self.dtype_to_last_index[dt] = self.index_count
        self.locations = list(locations_counts.keys())
        self.locations_probs = np.array([locations_counts[l] for l in self.locations]) / sum(locations_counts.values())
        self.index_to_node = {node.index: node for node in self.nodes}
        self.name_to_node: Dict[str, NodeData] = {node.name: node for node in self.nodes}

        mulecules = [node for node in self.nodes if node.type == NodeTypes.molecule]
        self.molecule_indexes = [node.index for node in mulecules]
        self.molecule_array = np.array([node.vec for node in mulecules])

        proteins = [node for node in self.nodes if node.type == NodeTypes.protein]
        self.protein_indexes = [node.index for node in proteins]
        self.protein_array = np.array([node.vec for node in proteins])

        texts = [node for node in self.nodes if node.type == NodeTypes.text]
        self.text_indexes = [node.index for node in texts]
        self.text_array = np.array([node.vec for node in texts])
        self.type_to_indexes={PROTEIN: self.protein_indexes, MOLECULE: self.molecule_indexes, TEXT: self.text_indexes}

    @lru_cache(maxsize=4000)
    def get_probs(self, what, index):
        node = self.index_to_node[index]
        if what == 'molecule':
            array_ = self.molecule_array
        elif what == 'protein':
            array_ = self.protein_array
        elif what == 'text':
            array_ = self.text_array
        else:
            raise ValueError(f'Unknown type {what}')
        distances = pairwise_distances(node.vec.reshape(1, -1), array_,
                                       metric='cosine').flatten()

        prob = np.exp(-distances)
        prob[distances == 0] = 0  # same element
        prob /= np.sum(prob)
        return prob

    def sample_entity(self, index, how='similar', what='molecule', k_closest=15):

        if what == 'molecule':
            entities = self.molecule_indexes
        elif what == 'protein':
            entities = self.protein_indexes
        elif what == 'text':
            entities = self.text_indexes
        else:
            raise ValueError(f'Unknown type {what}')
        good_index = False
        while not good_index:

            if how == 'similar':
                prob = self.get_probs(what=what, index=index)
                closest_index = np.argsort(prob)[::-1][:k_closest]
                closest_prob = prob[closest_index]
                closest_prob = closest_prob / np.sum(closest_prob)
                selected_index = random.choices(np.array(entities)[closest_index], closest_prob)[0]
            else:
                selected_index = random.choice([x for x in entities if x != index])

            if self.index_to_node[selected_index].have_seq:
                good_index = True
        return selected_index

    def sample_random_locations_map(self):
        mapping = {}
        for l in self.locations:
            mapping[l] = random.choice([x for x in self.locations if x != l])
        return {l: random.choices(self.locations, self.locations_probs) for l in self.locations}


def get_from_args(args):
    return NodesIndexManager(pretrained_method=args.gnn_pretrained_method, fuse_name=args.fuse_name,
                             fuse_pretrained_start=args.fuse_pretrained_start)


if __name__ == "__main__":
    node_index_manager = NodesIndexManager(pretrained_method=PRETRAINED_EMD_FUSE, fuse_name="all_to_one")
    print(node_index_manager.index_to_node[10].vec.shape)
