import dataclasses
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import seaborn as sns
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
labels_names = ReactionTag().get_names()

model, nodes_index_manager = load_model(learned_embedding_dim=256, hidden_channels=256, num_layers=3, out_channels=5,
                                        model_path="/home/amitay/PycharmProjects/reactome/data/model/model_mlc_256_2.pt", )
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
with plt.style.context('tableau-colorblind10', after_reset=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

real_values = []
predict_values = []
for data in tqdm(dataset):
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    out_prob = torch.sigmoid(out).detach().cpu().numpy()[0]
    predict_values.append(out_prob.tolist())
    real_values.append(data.tags.numpy().tolist())
real_values = np.array(real_values)
predict_values = np.array(predict_values)
for i, name in enumerate(labels_names):
    fpr, tpr, thresholds = roc_curve(real_values[:, i], predict_values[:, i])
    with plt.style.context('tableau-colorblind10', after_reset=True):
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})", color=colors[i])
with plt.style.context('tableau-colorblind10', after_reset=True):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')
    plt.legend()
    plt.savefig("../data/fig/roc_curve_mlc.png", dpi=300)
    plt.show()
