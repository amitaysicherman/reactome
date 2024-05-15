from model_tags import PartialFixedEmbedding, HeteroGNN
from index_manger import NodesIndexManager, get_edges_values, NodeTypes
from dataset_builder import ReactionDataset
from tagging import ReactionTag
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
import wandb

# run = wandb.init(id=run_id, project="reactome-tags")

learned_embedding_dim = 256
hidden_channels = 256
num_layers = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nodes_index_manager = NodesIndexManager()
root = "data/items"
layer_type = "SAGEConv"

model = HeteroGNN(nodes_index_manager, hidden_channels=hidden_channels, out_channels=1,
                  num_layers=num_layers,
                  learned_embedding_dim=learned_embedding_dim, train_all_emd=False, save_activation=True,
                  conv_type=layer_type).to(device)
model.load_state_dict(torch.load("/home/amitay/PycharmProjects/reactome/data/model/model_fake_256_28.pt", map_location=torch.device('cpu')))
model.eval()


@dataclass
class config:
    sample: int = 200
    entities: int = 0
    location: int = 0
    only_fake: bool = False
    title: str = ""


def plot_act(dataset, title, method="PCA"):
    loss = []
    for data in tqdm(dataset):
        x_dict = {key: data.x_dict[key] for key in data.x_dict.keys()}
        y = data['tags'].to(device)
        y = y.float()[-1:]

        edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
        out = model(x_dict, edge_index_dict)
        loss.append(torch.nn.BCEWithLogitsLoss()(out, y.unsqueeze(0)).item())
    print(title, np.mean(loss))
    model.plot_activations(title, reduce_dim_method=method, last_layer_only=False)


sample = 1000
configs = [
    config(sample=sample, entities=0, location=0, only_fake=False, title="FAKE"),
    # config(sample=sample, entities=0, location=1, only_fake=True, title="Fake Location"),
    # config(sample=sample, entities=1, location=0, only_fake=True, title="Fake Entities"),
    # config(sample=sample, entities=1, location=1, only_fake=True, title="Fake Entities and Location"),
    # config(sample=sample, entities=1, location=3, only_fake=False, title="Real_and_Fake"),
]
for config in configs:
    dataset = ReactionDataset(root=root, one_per_sample=True, sample=config.sample,
                              location_augmentation_factor=config.location,
                              molecule_similier_factor=config.entities,
                              molecule_random_factor=config.entities,
                              protein_similier_factor=config.entities, protein_random_factor=config.entities,
                              only_fake=config.only_fake).reactions
    plot_act(dataset, config.title, method="PCA")
    plot_act(dataset, config.title, method="TSNE")
name_to_file = {
    "Reaction PCA": "Real_and_Fake_PCA_complex_3_PCA.png",
    "Reaction TSNE": "Real_and_Fake_TSNE_complex_3_TSNE.png",
    "Complex PCA": "Real_and_Fake_PCA_complex_3_PCA.png",
    "Complex TSNE": "Real_and_Fake_TSNE_complex_3_TSNE.png",
}
# for key in name_to_file.keys():
#     wandb.log({key: wandb.Image(f'data/fig/{name_to_file[key]}')})
