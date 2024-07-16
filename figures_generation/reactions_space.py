import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import random
from dataset.index_manger import NodesIndexManager
from dataset.dataset_builder import get_reactions, get_reaction_entities_id_with_text
import numpy as np

PROTEIN = "protein"
MOLECULE = "molecule"


def plot_reaction_space(counter, fuse_model, prot_emd_type, mol_emd_type, pretrained_method=2):
    index_manager = NodesIndexManager(pretrained_method=pretrained_method, fuse_model=fuse_model,
                                      prot_emd_type=prot_emd_type, mol_emd_type=mol_emd_type)
    train_lines, val_lines, test_lines = get_reactions()
    reactions_ids = [792, 2940, 6942]

    ids = []
    types = []
    vecs = []
    for id_ in reactions_ids:
        reaction = train_lines[id_]
        names = get_reaction_entities_id_with_text(reaction, False)
        nodes = [index_manager.name_to_node[name] for name in names if
                 name in index_manager.name_to_node and (
                         index_manager.name_to_node[name].type in [PROTEIN, MOLECULE])]
        ids.extend([id_] * len(nodes))
        types.extend([node.type for node in nodes])
        vecs.extend([node.vec for node in nodes])

    type_to_shape = {'protein': 'o', 'molecule': 'X'}
    colors = ['red', 'blue', 'green', 'yellow']
    vecs = np.array(vecs)
    ids = np.array(ids)
    types = np.array(types)
    cosine_dist = cosine_distances(vecs)

    # Initialize and fit t-SNE with the precomputed cosine distance matrix
    tsne = TSNE(metric='precomputed', init="random", n_components=2, perplexity=4, n_iter=500, verbose=1)
    X_embedded = tsne.fit_transform(cosine_dist)

    # X_embedded = TSNE(n_components=2, perplexity=4).fit_transform(vecs)
    fig = plt.figure()
    for id_count, id_ in enumerate(np.unique(ids)):
        for type_ in np.unique(types):
            mask = (ids == id_) & (types == type_)
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], c=[colors[id_count]] * sum(mask),
                        marker=type_to_shape[type_], label=f'{type_}_{id_}')
    plt.legend()
    import os
    if not os.path.exists('data/figures/reactions_space'):
        os.makedirs('data/figures/reactions_space')
    plt.savefig(f'data/figures/reactions_space/{prot_emd_type}_{mol_emd_type}_{pretrained_method}_{counter}.png')
    plt.close(fig)
