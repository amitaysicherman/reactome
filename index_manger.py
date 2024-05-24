import numpy as np
import os
import dataclasses
from sklearn.metrics import pairwise_distances
import random
from collections import defaultdict
from typing import Dict
from matplotlib import pyplot as plt
from common import DATA_TYPES, EMBEDDING_DATA_TYPES, PROTEIN, DNA, MOLECULE, TEXT, LOCATION, UNKNOWN_ENTITY_TYPE, \
    REACTION, COMPLEX, TYPE_TO_VEC_DIM
from functools import lru_cache
import glob

REACTION_NODE_ID = 0
COMPLEX_NODE_ID = 1
UNKNOWN_ENTITY_TYPE = UNKNOWN_ENTITY_TYPE

NO_PRETRAINED_EMD = 0
PRETRAINED_EMD = 1
PRETRAINED_EMD_FUSE = 2
CONCAT_EMD = 3
pretrained_method_names = {
    NO_PRETRAINED_EMD: "no_pretrained",
    PRETRAINED_EMD: "pretrained",
    PRETRAINED_EMD_FUSE: "pretrained_fuse",
    CONCAT_EMD: "concat"
}


@dataclasses.dataclass
class NodeTypes:
    reaction = REACTION
    complex = COMPLEX
    location = LOCATION
    protein = PROTEIN
    dna = DNA
    molecule = MOLECULE
    text = TEXT


def get_node_types():
    return [NodeTypes.reaction, NodeTypes.complex, NodeTypes.location, NodeTypes.protein, NodeTypes.dna,
            NodeTypes.molecule, NodeTypes.text]


color_palette = plt.get_cmap("tab10")
node_colors = {node_type: color_palette(i) for i, node_type in enumerate(get_node_types())}


@dataclasses.dataclass
class EdgeTypes:
    # location
    location_self_loop = (NodeTypes.location, "location_self_loop", NodeTypes.location)
    location_to_protein = (NodeTypes.location, "location_to_protein", NodeTypes.protein)
    location_to_dna = (NodeTypes.location, "location_to_dna", NodeTypes.dna)
    location_to_molecule = (NodeTypes.location, "location_to_molecule", NodeTypes.molecule)

    # modification
    modification_self_loop = (NodeTypes.text, "modification_self_loop", NodeTypes.text)
    modification_to_protein = (NodeTypes.text, "modification_to_protein", NodeTypes.protein)

    # catalysis_activity
    catalysis_activity_self_loop = (NodeTypes.text, "catalysis_activity_self_loop", NodeTypes.text)
    catalysis_activity_to_reaction = (NodeTypes.text, "catalysis_activity_to_reaction", NodeTypes.reaction)

    # catalysis
    catalysis_protein_to_reaction = (NodeTypes.protein, "catalysis_protein_to_reaction", NodeTypes.reaction)
    catalysis_dna_to_reaction = (NodeTypes.dna, "catalysis_dna_to_reaction", NodeTypes.reaction)
    catalysis_molecule_to_reaction = (NodeTypes.molecule, "catalysis_molecule_to_reaction", NodeTypes.reaction)

    # reaction input
    protein_to_reaction = (NodeTypes.protein, "protein_to_reaction", NodeTypes.reaction)
    dna_to_reaction = (NodeTypes.dna, "dna_to_reaction", NodeTypes.reaction)
    molecule_to_reaction = (NodeTypes.molecule, "molecule_to_reaction", NodeTypes.reaction)

    # complex
    complex_to_reaction = (NodeTypes.complex, "complex_to_reaction", NodeTypes.reaction)
    protein_to_complex = (NodeTypes.protein, "protein_to_complex", NodeTypes.complex)
    dna_to_complex = (NodeTypes.dna, "dna_to_complex", NodeTypes.complex)
    molecule_to_complex = (NodeTypes.molecule, "molecule_to_complex", NodeTypes.complex)
    catalysis_protein_to_complex = (NodeTypes.protein, "catalysis_protein_to_complex", NodeTypes.complex)
    catalysis_dna_to_complex = (NodeTypes.dna, "catalysis_dna_to_complex", NodeTypes.complex)
    catalysis_molecule_to_complex = (NodeTypes.molecule, "catalysis_molecule_to_complex", NodeTypes.complex)

    # reaction output
    # reaction_to_protein = (NodeTypes.reaction, "reaction_to_protein", NodeTypes.protein)
    # reaction_to_dna = (NodeTypes.reaction, "reaction_to_dna", NodeTypes.dna)
    # reaction_to_molecule = (NodeTypes.reaction, "reaction_to_molecule", NodeTypes.molecule)

    # binding
    # protein_to_protein = (NodeTypes.protein, "protein_to_protein", NodeTypes.protein)
    # protein_to_molecule = (NodeTypes.protein, "protein_to_molecule", NodeTypes.molecule)
    # protein_to_dna = (NodeTypes.protein, "protein_to_dna", NodeTypes.dna)
    # molecule_to_protein = (NodeTypes.molecule, "molecule_to_protein", NodeTypes.protein)
    # molecule_to_molecule = (NodeTypes.molecule, "molecule_to_molecule", NodeTypes.molecule)
    # molecule_to_dna = (NodeTypes.molecule, "molecule_to_dna", NodeTypes.dna)
    # dna_to_protein = (NodeTypes.dna, "dna_to_protein", NodeTypes.protein)
    # dna_to_molecule = (NodeTypes.dna, "dna_to_molecule", NodeTypes.molecule)
    # dna_to_dna = (NodeTypes.dna, "dna_to_dna", NodeTypes.dna)

    def get(self, str):
        return getattr(self, str)

    def get_by_src_dst(self, src, dst, is_catalysis=False):
        if src == NodeTypes.text:
            if dst == NodeTypes.text:
                if is_catalysis:
                    text = f'catalysis_activity_self_loop'
                else:
                    text = f'modification_self_loop'
            elif is_catalysis:
                text = f'catalysis_activity_to_{dst}'
            else:
                text = f'modification_to_{dst}'
        elif src == dst and src == NodeTypes.location:
            text = f'location_self_loop'
        elif is_catalysis:
            text = f'catalysis_{src}_to_{dst}'
        else:
            text = f'{src}_to_{dst}'
        return self.get(text)


def get_edges_values():
    attributes = dir(EdgeTypes)
    edges = []
    for attr in attributes:
        value = getattr(EdgeTypes, attr)
        if isinstance(value, tuple) and len(value) == 3:
            edges.append(value)
    return edges


class NodeData:
    def __init__(self, index, name, type_, vec=None):
        self.index = index
        self.name = name
        self.type = type_
        self.vec = vec


def get_fuse_file_from_conf(root, config_name, dt):
    if config_name == "":
        if os.path.exists(f'{root}/{dt}_vec_fuse.npy'):
            return f'{root}/{dt}_vec_fuse.npy'
        else:
            print(f'fuse file found in {root}/{dt}_vec_fuse.npy')
            return f'{root}/{dt}_vec.npy'
    files_opt = glob.glob(f'{root}/{config_name}/{dt}*')
    if len(files_opt) == 0:
        print(f'No files found in {root}/{config_name}/{dt}*')
        return f'{root}/{dt}_vec.npy'
    epoch_to_file = {int(file_name.split('_')[-1].split('.')[0]): file_name for file_name in files_opt}
    return epoch_to_file[max(epoch_to_file.keys())]


class NodesIndexManager:
    def __init__(self, root="data/items", fuse_vec=PRETRAINED_EMD, fuse_config=""):
        reaction_node = NodeData(REACTION_NODE_ID, REACTION, NodeTypes.reaction)
        complex_node = NodeData(COMPLEX_NODE_ID, COMPLEX, NodeTypes.complex)
        self.nodes = [reaction_node, complex_node]
        locations_counts = {}
        self.index_count = 2
        self.dtype_to_first_index = dict()
        self.dtype_to_last_index = dict()
        for dt in DATA_TYPES:
            self.dtype_to_first_index[dt] = self.index_count
            names_file = f'{root}/{dt}.txt'
            with open(names_file) as f:
                lines = f.read().splitlines()
            if dt in EMBEDDING_DATA_TYPES:
                if fuse_vec == NO_PRETRAINED_EMD:
                    random.seed(42)
                    vectors = np.stack([np.random.rand(TYPE_TO_VEC_DIM[dt]) for _ in range(len(lines))])
                else:
                    fuse_vec_file = get_fuse_file_from_conf(root, fuse_config, dt)
                    print("fuse_vec_file", fuse_vec_file)
                    pretrained_vec_file = f'{root}/{dt}_vec.npy'

                    if fuse_vec == PRETRAINED_EMD_FUSE:
                        vectors = np.load(fuse_vec_file)
                    elif fuse_vec == PRETRAINED_EMD:
                        vectors = np.load(pretrained_vec_file)
                    else:  # CONCAT_EMD
                        vectors = np.concatenate([np.load(fuse_vec_file), np.load(pretrained_vec_file)], axis=1)

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
    n = NodesIndexManager()
    n.sample_entity(10_000, how='similar', what='molecule')
