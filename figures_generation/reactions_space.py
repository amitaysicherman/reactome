import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import random

from dataset.index_manger import NodesIndexManager
from dataset.dataset_builder import get_reactions, get_reaction_entities
import numpy as np
import os
from sklearn.decomposition import PCA
from common.path_manager import figures_path

PROTEIN = "protein"
MOLECULE = "molecule"
ENTITIES = [PROTEIN, MOLECULE]

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def plot_reaction_space(counter, fuse_model, prot_emd_type, mol_emd_type, pretrained_method=2,
                        reactions_ids=[100, 200]):
    index_manager = NodesIndexManager(pretrained_method=pretrained_method, fuse_model=fuse_model,
                                      prot_emd_type=prot_emd_type, mol_emd_type=mol_emd_type)

    train_lines, val_lines, test_lines = get_reactions(filter_dna=True)
    reactions = train_lines + val_lines + test_lines
    all_nodes = [node for node in index_manager.nodes if node.type in ENTITIES]
    reactoins_to_indexes = dict()
    for reaction_id in reactions_ids:
        reaction = reactions[reaction_id]
        entities = get_reaction_entities(reaction, True)
        names = [e.get_db_identifier() for e in entities]
        indexes = [index_manager.name_to_node[name].index for name in names if name in index_manager.name_to_node]
        reactoins_to_indexes[reaction.name] = indexes
    all_reaction_indexes = set(sum(reactoins_to_indexes.values(), []))
    nodes_in_reactions = [node for node in all_nodes if node.index in all_reaction_indexes]
    node_index_to_array_index = {node.index: i for i, node in enumerate(nodes_in_reactions)}
    all_vecs = np.array([node.vec for node in nodes_in_reactions])

    cosine_dist = cosine_distances(all_vecs)
    tsne = TSNE(metric='precomputed', init="random", n_components=2, perplexity=3, n_iter=500,
                verbose=0)
    X_embedded = tsne.fit_transform(cosine_dist)
    for i, (name, ids) in enumerate(reactoins_to_indexes.items()):
        reaction_nodes = [index_manager.index_to_node[id_] for id_ in ids]
        molecule_mask = [node.type == MOLECULE for node in reaction_nodes]
        protein_mask = [node.type == PROTEIN for node in reaction_nodes]
        reaction_x = X_embedded[[node_index_to_array_index[id_] for id_ in ids]]
        plt.scatter(reaction_x[molecule_mask][:, 0], reaction_x[molecule_mask][:, 1], c=COLORS[i], marker='+', s=50,
                    label=f'(Molecule){name}')
        plt.scatter(reaction_x[protein_mask][:, 0], reaction_x[protein_mask][:, 1], c=COLORS[i], marker='x', s=50,
                    label=f'(Protein){name}')

        for j, id_ in enumerate(ids):
            plt.text(reaction_x[j, 0], reaction_x[j, 1], index_manager.index_to_node[id_].name, fontsize=8)

        molecule_mask = [node.type == MOLECULE for node in nodes_in_reactions]
        protein_mask = [node.type == PROTEIN for node in nodes_in_reactions]

        mol_no_reactions = X_embedded[molecule_mask]
        prot_no_reactions = X_embedded[protein_mask]
        plt.scatter(mol_no_reactions[:, 0], mol_no_reactions[:, 1], c='gray', marker='+', s=5, label='Molecule')
        plt.scatter(prot_no_reactions[:, 0], prot_no_reactions[:, 1], c='gray', marker='x', s=5, label='Protein')

    plt.legend()
    plt.savefig(f'{figures_path}/reactions_space_{counter}.png')
    plt.show()
