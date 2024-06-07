import numpy as np
from sklearn.metrics import pairwise_distances
import random
from typing import Dict
from common.utils import TYPE_TO_VEC_DIM, load_fuse_model
from common.path_manager import item_path
from common.data_types import REACTION, COMPLEX, UNKNOWN_ENTITY_TYPE, PROTEIN, EMBEDDING_DATA_TYPES, LOCATION, \
    DATA_TYPES, NodeTypes, BIOLOGICAL_PROCESS
from model.models import apply_model
from functools import lru_cache

REACTION_NODE_ID = 0
COMPLEX_NODE_ID = 1
UNKNOWN_ENTITY_TYPE = UNKNOWN_ENTITY_TYPE

NO_PRETRAINED_EMD = 0
PRETRAINED_EMD = 1
PRETRAINED_EMD_FUSE = 2
pretrained_method_names = {
    NO_PRETRAINED_EMD: "no_pretrained",
    PRETRAINED_EMD: "pretrained",
    PRETRAINED_EMD_FUSE: "pretrained_fuse",
}

class NodeData:
    def __init__(self, index, name, type_, vec=None):
        self.index = index
        self.name = name
        self.type = type_
        self.vec = vec


class NodesIndexManager:
    def __init__(self, pretrained_method=PRETRAINED_EMD, fuse_name=""):
        reaction_node = NodeData(REACTION_NODE_ID, REACTION, NodeTypes.reaction)
        complex_node = NodeData(COMPLEX_NODE_ID, COMPLEX, NodeTypes.complex)
        self.nodes = [reaction_node, complex_node]
        locations_counts = {}
        self.index_count = 2
        self.dtype_to_first_index = dict()
        self.dtype_to_last_index = dict()
        self.bp_name_to_index = {"": -1}
        self.fuse_model = load_fuse_model(fuse_name)
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
                    random.seed(42)
                    vectors = np.stack([np.random.rand(TYPE_TO_VEC_DIM[dt]) for _ in range(len(lines))])
                else:
                    pretrained_vec_file = f'{item_path}/{dt}_vec.npy'
                    vectors = np.load(pretrained_vec_file)
                    if pretrained_method == PRETRAINED_EMD_FUSE:
                        vectors = apply_model(self.fuse_model, vectors, dt).detach().cpu().numpy()
            elif dt == UNKNOWN_ENTITY_TYPE:
                vectors = [np.zeros(TYPE_TO_VEC_DIM[PROTEIN]) for _ in range(len(lines))]
            else:
                vectors = [None] * len(lines)
            for i, line in enumerate(lines):
                name = "@".join(line.split("@")[:-1])
                node = NodeData(self.index_count, name, dt, vectors[i])
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

    @lru_cache(maxsize=4000)
    def get_probs(self, what, index):
        node = self.index_to_node[index]
        if what == 'molecule':
            array_ = self.molecule_array
        else:
            array_ = self.protein_array

        distances = pairwise_distances(node.vec.reshape(1, -1), array_,
                                       metric='cosine').flatten()

        prob = np.exp(-distances)
        prob[distances == 0] = 0  # same element
        prob /= np.sum(prob)
        return prob

    def sample_entity(self, index, how='similar', what='molecule', k_closest=15):
        if what == 'molecule':
            entities = self.molecule_indexes
        else:
            entities = self.protein_indexes

        if how == 'similar':
            prob = self.get_probs(what=what, index=index)
            closest_index = np.argsort(prob)[::-1][:k_closest]
            closest_prob = prob[closest_index]
            closest_prob = closest_prob / np.sum(closest_prob)
            return random.choices(np.array(entities)[closest_index], closest_prob)[0]
        else:
            return random.choice([x for x in entities if x != index])

    def sample_random_locations_map(self):
        mapping = {}
        for l in self.locations:
            mapping[l] = random.choice([x for x in self.locations if x != l])
        return {l: random.choices(self.locations, self.locations_probs) for l in self.locations}


if __name__ == "__main__":
    node_index_manager = NodesIndexManager(pretrained_method=PRETRAINED_EMD_FUSE, fuse_name="all_to_one")
    print(node_index_manager.index_to_node[10].vec.shape)
