import numpy as np
import os
import dataclasses
from sklearn.metrics import pairwise_distances
import random
from collections import defaultdict
REACTION_NODE_ID = 0


@dataclasses.dataclass
class NodeTypes:
    reaction = "reaction"
    protein = "protein"
    dna = "dna"
    molecule = "molecule"
    text = "text"
    location = "location"


def get_types_values():
    return [NodeTypes.reaction, NodeTypes.protein, NodeTypes.dna, NodeTypes.molecule, NodeTypes.text,
            NodeTypes.location]


node_colors = {
    NodeTypes.reaction: "pink",
    NodeTypes.protein: "blue",
    NodeTypes.dna: "green",
    NodeTypes.molecule: "yellow",
    NodeTypes.text: "orange",
    NodeTypes.location: "purple"
}


@dataclasses.dataclass
class EdgeTypes:
    location_self_loop = (NodeTypes.location, "location_self_loop", NodeTypes.location)
    location_to_protein = (NodeTypes.location, "location_to_protein", NodeTypes.protein)
    location_to_dna = (NodeTypes.location, "location_to_dna", NodeTypes.dna)
    location_to_molecule = (NodeTypes.location, "location_to_molecule", NodeTypes.molecule)
    modification_self_loop = (NodeTypes.text, "modification_self_loop", NodeTypes.text)
    modification_to_protein = (NodeTypes.text, "modification_to_protein", NodeTypes.protein)
    catalysis_activity_to_reaction = (NodeTypes.text, "catalysis_activity_to_reaction", NodeTypes.reaction)
    catalysis_activity_self_loop = (NodeTypes.text, "catalysis_activity_self_loop", NodeTypes.text)
    catalysis_protein_to_reaction = (NodeTypes.protein, "catalysis_protein_to_reaction", NodeTypes.reaction)
    catalysis_dna_to_reaction = (NodeTypes.dna, "catalysis_dna_to_reaction", NodeTypes.reaction)
    catalysis_molecule_to_reaction = (NodeTypes.molecule, "catalysis_molecule_to_reaction", NodeTypes.reaction)
    protein_to_reaction = (NodeTypes.protein, "protein_to_reaction", NodeTypes.reaction)
    dna_to_reaction = (NodeTypes.dna, "dna_to_reaction", NodeTypes.reaction)
    molecule_to_reaction = (NodeTypes.molecule, "molecule_to_reaction", NodeTypes.reaction)
    reaction_to_protein = (NodeTypes.reaction, "reaction_to_protein", NodeTypes.protein)
    reaction_to_dna = (NodeTypes.reaction, "reaction_to_dna", NodeTypes.dna)
    reaction_to_molecule = (NodeTypes.reaction, "reaction_to_molecule", NodeTypes.molecule)
    protein_to_protein = (NodeTypes.protein, "protein_to_protein", NodeTypes.protein)
    protein_to_molecule = (NodeTypes.protein, "protein_to_molecule", NodeTypes.molecule)
    protein_to_dna = (NodeTypes.protein, "protein_to_dna", NodeTypes.dna)
    molecule_to_protein = (NodeTypes.molecule, "molecule_to_protein", NodeTypes.protein)
    molecule_to_molecule = (NodeTypes.molecule, "molecule_to_molecule", NodeTypes.molecule)
    molecule_to_dna = (NodeTypes.molecule, "molecule_to_dna", NodeTypes.dna)
    dna_to_protein = (NodeTypes.dna, "dna_to_protein", NodeTypes.protein)
    dna_to_molecule = (NodeTypes.dna, "dna_to_molecule", NodeTypes.molecule)
    dna_to_dna = (NodeTypes.dna, "dna_to_dna", NodeTypes.dna)

    def get(self, str):
        return getattr(self, str)


def get_edges_values():
    return [EdgeTypes.location_self_loop, EdgeTypes.location_to_protein, EdgeTypes.location_to_dna,
            EdgeTypes.location_to_molecule, EdgeTypes.modification_self_loop, EdgeTypes.modification_to_protein,
            EdgeTypes.catalysis_activity_to_reaction, EdgeTypes.catalysis_activity_self_loop,
            EdgeTypes.catalysis_protein_to_reaction, EdgeTypes.catalysis_dna_to_reaction,
            EdgeTypes.catalysis_molecule_to_reaction, EdgeTypes.protein_to_reaction, EdgeTypes.dna_to_reaction,
            EdgeTypes.molecule_to_reaction, EdgeTypes.reaction_to_protein, EdgeTypes.reaction_to_dna,
            EdgeTypes.reaction_to_molecule, EdgeTypes.protein_to_protein, EdgeTypes.protein_to_molecule,
            EdgeTypes.protein_to_dna, EdgeTypes.molecule_to_protein, EdgeTypes.molecule_to_molecule,
            EdgeTypes.molecule_to_dna, EdgeTypes.dna_to_protein, EdgeTypes.dna_to_molecule, EdgeTypes.dna_to_dna]


def get_entity_reaction_type(type_, is_input, is_cat):
    if is_cat:
        assert is_input
    if is_cat:
        if type_ == NodeTypes.protein:
            return EdgeTypes.catalysis_protein_to_reaction
        elif type_ == NodeTypes.dna:
            return EdgeTypes.catalysis_dna_to_reaction
        elif type_ == NodeTypes.molecule:
            return EdgeTypes.catalysis_molecule_to_reaction
    if is_input:
        if type_ == NodeTypes.protein:
            return EdgeTypes.protein_to_reaction
        elif type_ == NodeTypes.dna:
            return EdgeTypes.dna_to_reaction
        elif type_ == NodeTypes.molecule:
            return EdgeTypes.molecule_to_reaction
    else:
        if type_ == NodeTypes.protein:
            return EdgeTypes.reaction_to_protein
        elif type_ == NodeTypes.dna:
            return EdgeTypes.reaction_to_dna
        elif type_ == NodeTypes.molecule:
            return EdgeTypes.reaction_to_molecule
    return False


def get_location_to_entity(type_):
    if type_ == NodeTypes.protein:
        return EdgeTypes.location_to_protein
    elif type_ == NodeTypes.dna:
        return EdgeTypes.location_to_dna
    elif type_ == NodeTypes.molecule:
        return EdgeTypes.location_to_molecule


def bind_edge(type1, type2):
    if type1 == NodeTypes.protein and type2 == NodeTypes.protein:
        return EdgeTypes.protein_to_protein
    elif type1 == NodeTypes.protein and type2 == NodeTypes.molecule:
        return EdgeTypes.protein_to_molecule
    elif type1 == NodeTypes.protein and type2 == NodeTypes.dna:
        return EdgeTypes.protein_to_dna
    elif type1 == NodeTypes.molecule and type2 == NodeTypes.protein:
        return EdgeTypes.molecule_to_protein
    elif type1 == NodeTypes.molecule and type2 == NodeTypes.molecule:
        return EdgeTypes.molecule_to_molecule
    elif type1 == NodeTypes.molecule and type2 == NodeTypes.dna:
        return EdgeTypes.molecule_to_dna
    elif type1 == NodeTypes.dna and type2 == NodeTypes.protein:
        return EdgeTypes.dna_to_protein
    elif type1 == NodeTypes.dna and type2 == NodeTypes.molecule:
        return EdgeTypes.dna_to_molecule
    elif type1 == NodeTypes.dna and type2 == NodeTypes.dna:
        return EdgeTypes.dna_to_dna
    else:
        raise ValueError(f"Unknown edge type: {type1} to {type2}")


class NodesIndexManager:
    def __init__(self, root="data/items", load_vectors=True):
        self.index_to_node = {REACTION_NODE_ID: NodeTypes.reaction}
        self.node_to_index = {NodeTypes.reaction: REACTION_NODE_ID}
        self.node_to_type = {REACTION_NODE_ID: NodeTypes.reaction}
        self.vector = {REACTION_NODE_ID: None}
        index = 1
        self.type_to_vectores = defaultdict(list)
        self.type_to_index = defaultdict(list)
        for name in ["entities", "locations", "catalyst_activities", "modifications"]:
            with open(f'{root}/{name}.txt') as f:
                lines = f.read().splitlines()
            if name == "entities":
                with open(f'{root}/{name}_sequences.txt') as f:
                    seqs = f.read().splitlines()
                    types = [seq.split(" ")[0] for seq in seqs]
            if load_vectors:
                vec_file = f'{root}/{name}_vec.npy'
                if os.path.exists(vec_file):
                    vecs = np.load(vec_file)
                    for i, vec in enumerate(vecs):
                        self.vector.update({index + i: vec})
                else:
                    self.vector.update({index: None for index in range(index, index + len(lines))})
            for line_index, line in enumerate(lines):
                *node, count = line.split(":")
                node = ":".join(node)
                self.index_to_node[index] = node
                self.node_to_index[node] = index
                if name in ["modifications", "catalyst_activities"]:
                    self.node_to_type[node] = NodeTypes.text
                elif name == "entities":
                    self.node_to_type[node] = types[line_index].lower()
                elif name == "locations":
                    self.node_to_type[node] = NodeTypes.location
                else:
                    self.node_to_type[node] = NodeTypes.reaction
                self.type_to_vectores[self.node_to_type[node]].append(self.vector[index])
                self.type_to_index[self.node_to_type[node]].append(index)
                index += 1

    def get_node(self, index):
        if index not in self.index_to_node:
            return None
        return self.index_to_node[index]

    def get_index(self, node):
        if node not in self.node_to_index:
            return None
        return self.node_to_index[node]

    def get_type(self, node):
        if node not in self.node_to_type:
            return None
        return self.node_to_type[node].lower()
    def get_index_type(self, index):
        return self.get_type(self.get_node(index))

    def sample_closest_n_from_k(self, index, n=50, k=5):
        type_ = self.get_type(index)
        all_options = self.type_to_vectores[type_]
        all_options_index = np.array(self.type_to_index[type_])
        vec = np.array(self.vector[self.get_index(index)])
        distances = pairwise_distances(vec.reshape(1, -1), np.array(all_options), metric='euclidean').flatten()
        closest_indices = np.argsort(distances)[:n]
        all_options_index = all_options_index[closest_indices]
        return random.sample(list(all_options_index), k=k)

if __name__ == "__main__":
    n = NodesIndexManager(load_vectors=True)
