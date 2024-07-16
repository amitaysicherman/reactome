from dataset.index_manger import NodesIndexManager
from common.data_types import NodeTypes
import torch
import numpy as np
import seaborn as sns
from dataset.dataset_builder import get_reactions, get_reaction_entities
from common.path_manager import figures_path
import matplotlib.pyplot as plt
from dataset.fuse_dataset import PairsDataset

nodes_index_manager = NodesIndexManager()

total_proteins = len([node for node in nodes_index_manager.nodes if node.type == NodeTypes.protein])
total_molecules = len([node for node in nodes_index_manager.nodes if node.type == NodeTypes.molecule])
print(f"Total proteins: {total_proteins}")
print(f"Total molecules: {total_molecules}")

sns.set_theme(style="white")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reactions = get_reactions(filter_unknown=False, filter_dna=False, filter_no_seq=False, filter_untrain=False,
                          filter_no_act=False, filter_no_mol=False, filter_singal_entity=False)
reactions = sum(reactions, [])  # train_lines + val_lines + test_lines
print("Number of reactions:", len(reactions))
knowns_reactions = get_reactions(filter_unknown=True, filter_dna=True, filter_no_seq=True, filter_untrain=False,
                                 filter_no_act=False, filter_no_mol=False, filter_singal_entity=True)
knowns_reactions = sum(knowns_reactions, [])
print("Number of known reactions:", len(knowns_reactions))

pairs_dataset = PairsDataset(knowns_reactions, nodes_index_manager)
print("Number of pairs:", len(pairs_dataset.all_pairs), len(pairs_dataset.pairs_unique))

proteins_per_reaction = []
mol_per_reaction = []
for reaction in knowns_reactions:
    p_reaction = set()
    m_reaction = set()
    entities = get_reaction_entities(reaction, True)
    for entity in entities:
        name = entity.get_db_identifier()
        if name in nodes_index_manager.name_to_node:
            node = nodes_index_manager.name_to_node[name]
            if node.type == NodeTypes.protein:
                p_reaction.add(node.index)
            elif node.type == NodeTypes.molecule:
                m_reaction.add(node.index)
    proteins_per_reaction.append(len(p_reaction))
    mol_per_reaction.append(len(m_reaction))
print(
    f"Proteins per reaction: {np.mean(proteins_per_reaction):.2f} ± {np.std(proteins_per_reaction):.2f} Median {np.quantile(proteins_per_reaction, 0.5)}")
print(
    f"Molecules per reaction: {np.mean(mol_per_reaction):.2f} ± {np.std(mol_per_reaction):.2f} Median {np.quantile(mol_per_reaction, 0.5)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.hist(proteins_per_reaction, bins=range(0, 20), color='skyblue', edgecolor='black', linewidth=1.2)
ax1.set_title("Proteins per reaction")
ax2.hist(mol_per_reaction, bins=range(0, 10), color='skyblue', edgecolor='black', linewidth=1.2)
ax2.set_title("Molecules per reaction")
plt.tight_layout()
plt.savefig(f"{figures_path}/proteins_molecules_per_reaction.png")
plt.show()
