import dataclasses
import random

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from biopax_parser import reaction_from_str, Reaction
from dataset_builder import get_nx_for_tags_prediction_task, nx_to_torch_geometric, have_unkown_nodes, \
    replace_location_augmentation, replace_entity_augmentation, add_if_not_none
from index_manger import NodesIndexManager, NodeTypes
from model_tags import HeteroGNN
from tagging import ReactionTag
from utils import reaction_to_data, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NT = NodeTypes()
root = "../data/items"
colors = sns.color_palette("tab10")

FAKE_TASK = False


def get_fake_data(data, nodes_index_manager: NodesIndexManager, type_):
    if type_ == "":
        return data
    elif type_ == "change_location":
        return replace_location_augmentation(nodes_index_manager, data)
    elif type_ == "change_random_protein":
        return replace_entity_augmentation(nodes_index_manager, data, NT.protein, "random")
    elif type_ == "change_random_molecule":
        return replace_entity_augmentation(nodes_index_manager, data, NT.molecule, "random")
    elif type_ == "change_similar_protein":
        return replace_entity_augmentation(nodes_index_manager, data, NT.protein, "similar")
    elif type_ == "change_similar_molecule":
        return replace_entity_augmentation(nodes_index_manager, data, NT.molecule, "similar")
    else:
        raise ValueError(f"Unknown type {type_}")


model, nodes_index_manager = load_model(
    model_path="/home/amitay/PycharmProjects/reactome/data/model/nofuse_model_fake_256_49.pt", learned_embedding_dim=256,
    hidden_channels=256, num_layers=3, out_channels=1, fuse=True)
with open(f'{root}/reaction.txt') as f:
    lines = f.readlines()
lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
reactions = [reaction_from_str(line) for line in tqdm(lines)]
reactions = reactions[int(len(reactions) * 0.8):]
dataset = []
for reaction in reactions:
    if have_unkown_nodes(reaction, nodes_index_manager):
        continue
    data = reaction_to_data(reaction, nodes_index_manager)
    if data is None or data.tags.sum().item() == 0:
        continue
    dataset.append(data)

all_dataset = {
    'real': dataset,
    'location': [get_fake_data(data, nodes_index_manager, "change_location") for data in dataset],
    'entities': [get_fake_data(data, nodes_index_manager, "change_random_protein") for data in dataset] + [
        get_fake_data(data, nodes_index_manager, "change_random_molecule") for data in dataset],
    # 'similar_protein': [get_fake_data(data, nodes_index_manager, "change_similar_protein") for data in dataset],
    # 'similar_molecule': [get_fake_data(data, nodes_index_manager, "change_similar_molecule") for data in dataset],
}
with plt.style.context('tableau-colorblind10', after_reset=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

for i, (name, data_type) in enumerate(all_dataset.items()):
    real_values = []
    predict_values = []
    if name == "real":
        continue
    data_type += all_dataset['real']

    for data in tqdm(data_type):
        if data is None:
            continue
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
        out_prob = torch.sigmoid(out).detach().cpu().numpy()[0]
        predict_values.append(out_prob.tolist())
        real_values.append(data.tags.numpy()[-1].tolist())

    real_values = np.array(real_values).flatten()
    predict_values = np.array(predict_values).flatten()
    fpr, tpr, thresholds = roc_curve(real_values, predict_values)
    print(name, auc(fpr, tpr))
    with plt.style.context('tableau-colorblind10', after_reset=True):

        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})", color=colors[i])
with plt.style.context('tableau-colorblind10', after_reset=True):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')
    plt.legend()
    plt.savefig("../data/fig/roc_curve_fake.png", dpi=300)
    plt.show()
