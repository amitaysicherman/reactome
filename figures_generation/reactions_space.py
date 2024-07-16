import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import random
from dataset.index_manger import NodesIndexManager
from dataset.dataset_builder import get_reactions, get_reaction_entities_id_with_text
import numpy as np
import os

PROTEIN = "protein"
MOLECULE = "molecule"


def plot_reaction_space(counter, fuse_model, prot_emd_type, mol_emd_type, pretrained_method=2):
    # Initialize index manager with given parameters
    index_manager = NodesIndexManager(pretrained_method=pretrained_method, fuse_model=fuse_model,
                                      prot_emd_type=prot_emd_type, mol_emd_type=mol_emd_type)

    # Get reaction data
    train_lines, val_lines, test_lines = get_reactions()

    # Define specific reaction IDs to analyze
    reactions_ids = [792, 2940, 6942]
    reactions_names = [train_lines[id_].name for id_ in reactions_ids]

    # Initialize containers for IDs, types, and vectors
    ids, types, vecs = [], [], []

    for id_ in reactions_ids:
        reaction = train_lines[id_]
        # Extract entities and their vectors
        names = get_reaction_entities_id_with_text(reaction, False)
        nodes = [index_manager.name_to_node[name] for name in names if
                 name in index_manager.name_to_node and
                 index_manager.name_to_node[name].type in [PROTEIN, MOLECULE]]

        # Extend containers with current reaction data
        ids.extend([id_] * len(nodes))
        types.extend([node.type for node in nodes])
        vecs.extend([node.vec for node in nodes])

    # Convert lists to numpy arrays
    vecs = np.array(vecs)
    ids = np.array(ids)
    types = np.array(types)

    # Compute cosine distance matrix
    cosine_dist = cosine_distances(vecs)

    # Initialize and fit t-SNE
    tsne = TSNE(metric='precomputed', init="random", n_components=2, perplexity=min(len(vecs),30), n_iter=500, verbose=1)
    X_embedded = tsne.fit_transform(cosine_dist)

    # Plot the embedded space without legend
    type_to_shape = {PROTEIN: 'o', MOLECULE: 'X'}
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    fig, ax = plt.subplots()
    for id_count, id_ in enumerate(np.unique(ids)):
        for type_ in np.unique(types):
            mask = (ids == id_) & (types == type_)
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], c=[colors[id_count]] * sum(mask),
                       marker=type_to_shape[type_], s=50, edgecolor='k',
                       label=f'{type_} | {reactions_names[reactions_ids.index(id_)]}')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # Save the plot without legend
    output_dir = 'data/figures/reactions_space'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{prot_emd_type}_{mol_emd_type}_{pretrained_method}_{counter}_no_legend.png',
                bbox_inches='tight')
    plt.close(fig)

    # Create a figure for the legend only
    fig_legend = plt.figure()
    fig_legend.legend(*ax.get_legend_handles_labels(), loc='center', fontsize='small', markerscale=1.2)

    # Save the legend as a separate plot
    plt.savefig(f'{output_dir}/{prot_emd_type}_{mol_emd_type}_{pretrained_method}_{counter}_legend.png',
                bbox_inches='tight')
    plt.close(fig_legend)
