import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import random
from dataset.index_manger import NodesIndexManager
from dataset.dataset_builder import get_reactions, get_reaction_entities
import numpy as np
import os
from sklearn.decomposition import PCA

PROTEIN = "protein"
MOLECULE = "molecule"
ENTITIES = [PROTEIN, MOLECULE]

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']

reactions_ids = [100, 200, 400]


def plot_reaction_space(counter, fuse_model, prot_emd_type, mol_emd_type, pretrained_method=2):
    # Initialize index manager with given parameters
    index_manager = NodesIndexManager(pretrained_method=pretrained_method, fuse_model=fuse_model,
                                      prot_emd_type=prot_emd_type, mol_emd_type=mol_emd_type)

    # Get reaction data

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
    nodes_no_reactions = [node for node in all_nodes if node.index not in all_reaction_indexes]
    nodes_in_reactions = [node for node in all_nodes if node.index in all_reaction_indexes]
    node_index_to_array_index = {node.index: i for i, node in enumerate(nodes_in_reactions)}
    all_vecs = np.array([node.vec for node in nodes_no_reactions + nodes_in_reactions])
    molecule_mask = [node.type == MOLECULE for node in nodes_no_reactions]
    protein_mask = [node.type == PROTEIN for node in nodes_no_reactions]

    cosine_dist = cosine_distances(all_vecs)
    tsne = TSNE(metric='precomputed', init="random", n_components=2, perplexity=3, n_iter=500,
                verbose=0)
    X_embedded = tsne.fit_transform(cosine_dist)

    X_embedded_no_reactions = X_embedded[:len(nodes_no_reactions)]

    mol_no_reactions = X_embedded_no_reactions[molecule_mask]
    prot_no_reactions = X_embedded_no_reactions[protein_mask]
    plt.scatter(mol_no_reactions[:, 0], mol_no_reactions[:, 1], c='gray', marker='+', s=5, label='Molecule')
    plt.scatter(prot_no_reactions[:, 0], prot_no_reactions[:, 1], c='gray', marker='x', s=5, label='Protein')
    for i, (name, ids) in enumerate(reactoins_to_indexes.items()):

        x = X_embedded[[node_index_to_array_index[id_] for id_ in ids]]
        plt.scatter(x[:, 0], x[:, 1], c=COLORS[i], marker='X', s=50, edgecolor='k', label=name)
    plt.legend()
    plt.show()
    #
    #     reactoins_to_indexes[reaction.name]=i
    # for id_ in reactions_ids:
    #
    # all_vecs_2d = TSNE(n_components=2).fit_transform(all_vecs)
    # all_types = [node.type for node in all_nodes]
    #
    # reactions_names = [train_lines[id_].name for id_ in reactions_ids]
    #
    # # Initialize containers for IDs, types, and vectors
    # ids, types, vecs = [], [], []
    #
    # for id_ in reactions_ids:
    #     reaction = train_lines[id_]
    #     # Extract entities and their vectors
    #     names = get_reaction_entities_id_with_text(reaction, False)
    #     nodes = [index_manager.name_to_node[name] for name in names if
    #              name in index_manager.name_to_node and
    #              index_manager.name_to_node[name].type in ENTITIES]
    #
    #     # Extend containers with current reaction data
    #     ids.extend([id_] * len(nodes))
    #     types.extend([node.type for node in nodes])
    #     vecs.extend([node.vec for node in nodes])
    #
    # # Convert lists to numpy arrays
    # vecs = np.array(vecs)
    # ids = np.array(ids)
    # types = np.array(types)
    #
    # # Compute cosine distance matrix
    # cosine_dist = (cosine_distances(vecs) * 50).astype(int)
    # names = [f"{id_}_{type_}" for id_, type_ in zip(ids, types)]
    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(cosine_dist, xticklabels=names, yticklabels=names, annot=True, ax=ax)
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig(f'data/figures/reactions_space/{prot_emd_type}_{mol_emd_type}_{pretrained_method}_{counter}.png')
    # # Initialize and fit t-SNE
    # tsne = TSNE(metric='precomputed', init="random", n_components=2, perplexity=3, n_iter=500,
    #             verbose=0)
    # X_embedded = tsne.fit_transform(cosine_dist)
    #
    # X_embedded = PCA(n_components=2).fit_transform(cosine_dist)
    #
    # # Plot the embedded space without legend
    # type_to_shape = {PROTEIN: 'o', MOLECULE: 'X'}
    # colors = ['#e41a1c', '#377eb8', '#4daf4a']
    #
    # fig, ax = plt.subplots()
    # for id_count, id_ in enumerate(np.unique(ids)):
    #     for type_ in np.unique(types):
    #         mask = (ids == id_) & (types == type_)
    #         ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], c=[colors[id_count]] * sum(mask),
    #                    marker=type_to_shape[type_], s=50, edgecolor='k',
    #                    label=f'{type_} | {reactions_names[reactions_ids.index(id_)]}')
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()
    # # # Save the plot without legend
    # # output_dir = 'data/figures/reactions_space'
    # # os.makedirs(output_dir, exist_ok=True)
    # # plt.savefig(f'{output_dir}/{prot_emd_type}_{mol_emd_type}_{pretrained_method}_{counter}_no_legend.png',
    # #             bbox_inches='tight')
    # # plt.close(fig)
    # #
    # # # Create a figure for the legend only
    # # fig_legend = plt.figure()
    # # fig_legend.legend(*ax.get_legend_handles_labels(), loc='center', fontsize='small', markerscale=1.2)
    # #
    # # # Save the legend as a separate plot
    # # plt.savefig(f'{output_dir}/{prot_emd_type}_{mol_emd_type}_{pretrained_method}_{counter}_legend.png',
    # #             bbox_inches='tight')
    # # plt.close(fig_legend)
