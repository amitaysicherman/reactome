from dataset.index_manger import NodesIndexManager
from common.data_types import NodeTypes
import torch
from model.tagging import ReactionTag
from dataset.dataset_builder import have_unkown_nodes, reaction_to_data, reaction_from_str
import dataclasses
import numpy as np
import seaborn as sns
from common.path_manager import item_path, figures_path
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

sns.set_theme(style="white")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tag_names = list(dataclasses.asdict(ReactionTag()).keys())

NT = NodeTypes()

nodes_index_manager = NodesIndexManager()

dtype_to_count = {}
for dtype in nodes_index_manager.dtype_to_first_index:
    with open(f"{item_path}/{dtype}.txt") as f:
        lines = f.read().splitlines()
    total_count = 0
    for line in lines:
        try:
            _, _, count = line.split("@")
            total_count += int(count)
        except:
            pass
    dtype_to_count[dtype] = total_count
    print(f"{dtype}: {total_count:,} total count")

print(f"Total nodes: {sum(dtype_to_count.values()):,}")

total_unique_elements = 0
for dtype in nodes_index_manager.dtype_to_first_index:
    unique_element = nodes_index_manager.dtype_to_last_index[dtype] - nodes_index_manager.dtype_to_first_index[dtype]
    print(f"{dtype}: {unique_element:,} unique elements")
    total_unique_elements += unique_element
print(f"Total unique elements: {total_unique_elements:,}")

with open(f'{item_path}/reaction.txt') as f:
    lines = f.readlines()
print(f"Number of reactions: {len(lines):,}")
dataset = []
no_tags_reaction = 0
for reaction in lines:
    if have_unkown_nodes(reaction_from_str(reaction), nodes_index_manager):
        no_tags_reaction += 1
        continue
    data = reaction_to_data(reaction, nodes_index_manager)
    if data is None or data.tags.sum().item() == 0:
        no_tags_reaction += 1
        continue
    dataset.append(data)
print(f"Number of reactions without tags: {no_tags_reaction:,}")
print(f"Number of reactions with tags: {len(dataset):,}")
tags = [data.tags.cpu().numpy().tolist() for data in dataset]
counts = pd.DataFrame(tags).sum(axis=0).values

# create pie chart
fig, ax = plt.subplots()
counts = counts[:-1]  # remove fake tag
tag_names = tag_names[:-1]  # remove fake tag
tag_names = [x.upper() for x in tag_names]
ax.pie(counts, labels=tag_names, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Labels Distribution")
plt.savefig(os.path.join(figures_path, "tags_distribution.png"), dpi=300)
plt.show()

# bar chart for the number of tags per reaction
tags_per_reaction = [data.tags.cpu().numpy().sum() for data in dataset]
tags_per_reaction = np.array(tags_per_reaction)
fig, ax = plt.subplots()
ax.bar(*np.unique(tags_per_reaction, return_counts=True))
plt.title("Number of labels per reaction")
plt.xlabel("Number of labels")
plt.ylabel("Number of reactions")

output_path = os.path.join(figures_path, "tags_per_reaction.png")
plt.savefig(output_path, dpi=300)
plt.show()

dtype_to_proccess_count = defaultdict(list)

for node in nodes_index_manager.index_to_node.values():
    vec = node.vec
if vec is not None:
    dtype_to_proccess_count[node.type].append(not np.allclose(vec, 0))
else:
    dtype_to_proccess_count[node.type].append(False)
for dtype in dtype_to_proccess_count:
    print(
        f"{dtype}: {np.mean(dtype_to_proccess_count[dtype])}, {np.sum(dtype_to_proccess_count[dtype])}, {len(dtype_to_proccess_count[dtype])}")
