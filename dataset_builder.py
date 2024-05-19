import dataclasses
import random

import index_manger
from index_manger import NodesIndexManager, EdgeTypes, NodeTypes, node_colors
from tagging import tag
from collections import defaultdict
from common import UNKNOWN_ENTITY_TYPE
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
from biopax_parser import reaction_from_str
import numpy as np
import json

ET = EdgeTypes()
NT = NodeTypes()

REACTION = NT.reaction
VIS = False
REAL = "real"
FAKE_LOCATION_ALL = "fake_location_all"
FAKE_LOCATION_SINGLE = "fake_location_single"
FAKE_PROTEIN = "fake_protein"
FAKE_MOLECULE = "fake_molecule"


def complex_index_to_node_index(complex_id):
    return f'c_{complex_id}'


def get_nx_for_tags_prediction_task(reaction, node_index_manager: NodesIndexManager):
    tags = tag(reaction)
    G = nx.DiGraph()
    reaction_node = node_index_manager.name_to_node[NT.reaction]
    G.add_node(reaction_node.index, name=reaction_node.name, type=reaction_node.type)
    complexes = set()
    catalysis_nodes = sum([c.entities for c in reaction.catalysis], [])
    for element in reaction.inputs + catalysis_nodes:
        complexes.add(complex_index_to_node_index(element.complex_id))
    for complex_id in complexes:
        if complex_id == complex_index_to_node_index(0):
            continue
        G.add_node(complex_id, name=NT.complex, type=NT.complex)
        G.add_edge(complex_id, reaction_node.index, type=ET.complex_to_reaction)

    is_cat_list = [False] * len(reaction.inputs) + [True] * len(catalysis_nodes)
    for element, is_cat in zip(reaction.inputs + catalysis_nodes, is_cat_list):
        node = node_index_manager.name_to_node[element.get_unique_id()]
        G.add_node(node.index, name=node.name, type=node.type)
        if element.complex_id == 0:
            edge_type = ET.get_by_src_dst(node.type, NT.reaction, is_catalysis=is_cat)
            G.add_edge(node.index, reaction_node.index, type=edge_type)
        else:
            edge_type = ET.get_by_src_dst(node.type, NT.complex, is_catalysis=is_cat)
            G.add_edge(node.index, complex_index_to_node_index(element.complex_id), type=edge_type)

        location_node = node_index_manager.name_to_node[element.location]
        if location_node.index not in G.nodes:
            G.add_node(location_node.index, name=location_node.name, type=NT.location)
            G.add_edge(location_node.index, location_node.index, type=ET.location_self_loop)
        G.add_edge(location_node.index, node.index, type=ET.get_by_src_dst(NT.location, node.type, is_catalysis=False))
        if node.type == NT.protein:
            for modification in element.modifications:
                modification_node = node_index_manager.name_to_node[f'TEXT@{modification}']
                if modification_node.index not in G.nodes:
                    G.add_node(modification_node.index, name=modification_node.name, type=NT.text)
                    G.add_edge(modification_node.index, modification_node.index, type=ET.modification_self_loop)
                G.add_edge(modification_node.index, node.index, type=ET.modification_to_protein)

    catalysis_activity_nodes = [node_index_manager.name_to_node[f'GO@{c.activity}'] for c in reaction.catalysis]

    for activity_node in catalysis_activity_nodes:
        if activity_node.index not in G.nodes:
            G.add_node(activity_node.index, name=activity_node.name, type=activity_node.type)
            G.add_edge(activity_node.index, activity_node.index, type=ET.catalysis_activity_self_loop)
        G.add_edge(activity_node.index, reaction_node.index, type=ET.catalysis_activity_to_reaction)
    return G, tags


def have_unkown_nodes(reaction, node_index_manager: NodesIndexManager, check_output=False):
    if check_output:
        entitites = reaction.inputs + reaction.outputs + sum([c.entities for c in reaction.catalysis], [])
    else:
        entitites = reaction.inputs + sum([c.entities for c in reaction.catalysis], [])
    for e in entitites:
        if node_index_manager.name_to_node[e.get_unique_id()].type == UNKNOWN_ENTITY_TYPE:
            return True
    return False


def build_dataset_for_tags_prediction_task(reaction: str, node_index_manager: NodesIndexManager, vis: bool = False):
    reaction = reaction_from_str(reaction)
    if have_unkown_nodes(reaction, node_index_manager):
        return None
    g, tags = get_nx_for_tags_prediction_task(reaction, node_index_manager)
    if vis:
        title = f"{reaction.name}\n{tags}"
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        plot_graph(g, node_index_manager, ax, title)
        plt.show()
    return nx_to_torch_geometric(g, tags=torch.Tensor(dataclasses.astuple(tags)).to(torch.float32),
                                 augmentation_type=REAL)


def plot_graph(G, node_index_manager, ax, title=""):
    # add order type for each node:
    order_map = {
        REACTION: 4,
        NT.complex: 3,
        NT.text: 2,
        NT.protein: 1,
        NT.molecule: 1,
        NT.dna: 1,
        "?": 1,
        NT.location: 0
    }
    for id, data in G.nodes(data=True):
        if data["type"] not in order_map:
            print(data["type"])
            data["type"] = "?"
        data["order"] = order_map[data["type"]]
    # pos = nx.spring_layout(G)
    pos = nx.multipartite_layout(G, subset_key="order")
    node_types = nx.get_node_attributes(G, "type")
    nodes_colors = [node_colors[node_type] for node_type in node_types.values()]
    nodes_labels = {n: data['name'] for n, data in G.nodes(data=True)}
    nx.draw_networkx_nodes(G, pos, node_size=2000, ax=ax,
                           node_color=nodes_colors, alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=1, ax=ax)
    nx.draw_networkx_labels(G, pos, nodes_labels, font_size=20, ax=ax)
    edge_labels = nx.get_edge_attributes(G, "type")
    for k, v in edge_labels.items():
        if v is None:
            edge_labels[k] = "?"
        else:
            edge_labels[k] = v[1]
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
    ax.set_title(title, fontsize=20)
    ax.axis("off")


def nx_to_torch_geometric(G: nx.Graph, **kwargs):
    hetero_graph = HeteroData()
    nodes_dict = defaultdict(list)
    edges_dict = defaultdict(list)
    nodes_index_to_hetro = dict()

    for node_id, node_data in G.nodes(data=True):
        node_type = node_data['type']
        nodes_index_to_hetro[node_id] = len(nodes_dict[node_type])
        nodes_dict[node_type].append(node_id)

    for node_type, node_ids in nodes_dict.items():
        if node_type == NT.complex:
            node_ids = [index_manger.COMPLEX_NODE_ID] * len(node_ids)
        hetero_graph[node_type].x = torch.LongTensor(node_ids).reshape(-1, 1)

    for src_id, dst_id, edge_data in G.edges(data=True):
        edge_type = edge_data["type"]
        edges_dict[edge_type].append([nodes_index_to_hetro[src_id], nodes_index_to_hetro[dst_id]])

    for edge_type, edge_list in edges_dict.items():
        if edge_type is None or edge_type == False:
            continue
        edges = torch.IntTensor(edge_list).t().to(torch.int64)
        hetero_graph[edge_type].edge_index = edges
    for k, v in kwargs.items():
        hetero_graph[k] = v
    try:
        hetero_graph.validate(raise_on_error=True)
        return hetero_graph

    except Exception as error:
        print("An exception occurred:", error)
        return None


def get_fake_tag(data):
    tags = torch.zeros_like(data.tags).to(torch.float32)
    tags[-1] = 1.0
    return tags


def clone_hetero_data(data: HeteroData, change_nodes_mapping=dict(), change_edge=False):
    new_hetero_data = HeteroData()
    for key, value in data.x_dict.items():
        new_hetero_data[key].x = torch.clone(value)

    for change_key, change_value in change_nodes_mapping.items():
        if change_edge:
            new_hetero_data[change_key].edge_index = torch.clone(data.edge_index_dict[change_key])
        else:
            new_hetero_data[change_key].x = change_value
    for key, value in data.edge_index_dict.items():
        new_hetero_data[key].edge_index = torch.clone(value)
    new_hetero_data.tags = get_fake_tag(data)
    return new_hetero_data


def replace_location_augmentation(index_manager: NodesIndexManager, data: HeteroData):
    if NodeTypes.location not in data.x_dict:
        return None
    random_mapping = index_manager.sample_random_locations_map()
    new_locations = [random_mapping[l.item()] for l in data.x_dict[NodeTypes.location]]
    change_nodes_mapping = {NodeTypes.location: torch.LongTensor(new_locations).reshape(-1, 1)}
    clone_data = clone_hetero_data(data, change_nodes_mapping)
    clone_data.augmentation_type = FAKE_LOCATION_ALL
    return clone_data


def replace_entity_location_augmentation(index_manager: NodesIndexManager, data: HeteroData):
    dtype = random.choice([NodeTypes.molecule, NodeTypes.protein])
    if dtype not in data.x_dict:
        return None
    new_index = [m.item() for m in data.x_dict[dtype]]
    change_index = random.choice(range(len(new_index)))

    locations_options = data.x_dict[NodeTypes.location].tolist()
    locations_options = [l[0] for l in locations_options]
    if len(locations_options) <= 1:
        return None

    edge_type = ET.get_by_src_dst(NodeTypes.location, dtype, is_catalysis=False)
    new_edges_index_type = torch.clone(data.edge_index_dict[edge_type])

    for i in range(new_edges_index_type.shape[1]):
        if new_edges_index_type[1][i] == change_index:
            options = [l for l in locations_options if l != new_edges_index_type[0][i].item()]
            new_edges_index_type[0][i] = random.choice(options)
    change_nodes_mapping = {edge_type: new_edges_index_type}
    clone_data = clone_hetero_data(data, change_nodes_mapping)
    clone_data.augmentation_type = FAKE_LOCATION_SINGLE
    return clone_data


def replace_entity_augmentation(index_manager: NodesIndexManager, data: HeteroData, dtype, how):
    if dtype not in data.x_dict:
        return None
    new_index = [m.item() for m in data.x_dict[dtype]]
    indexes_without_complexes = list(range(len(new_index)))
    change_index = random.choice(indexes_without_complexes)
    new_index[change_index] = index_manager.sample_entity(new_index[change_index], how=how, what=dtype)
    change_nodes_mapping = {dtype: torch.LongTensor(new_index).reshape(-1, 1)}
    clone_data = clone_hetero_data(data, change_nodes_mapping)
    if dtype == NodeTypes.molecule:
        clone_data.augmentation_type = FAKE_MOLECULE
    else:
        clone_data.augmentation_type = FAKE_PROTEIN
    return clone_data


def add_if_not_none(data, new_data):
    if new_data is not None:
        data.append(new_data)


class ReactionDataset:
    def __init__(self, root="data/items", sample=0, location_augmentation_factor=0, molecule_similier_factor=0,
                 molecule_random_factor=0, protein_similier_factor=0, protein_random_factor=0,
                 replace_entity_location=0, only_fake=False,
                 one_per_sample=False, order="date"):
        self.root = root
        self.node_index_manager = NodesIndexManager(root)
        self.reactions = []
        with open(f'{root}/reaction.txt') as f:
            lines = f.readlines()
        if order == "random":
            import random
            random.seed(42)
            random.shuffle(lines)
        elif order == "date":
            lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
        if sample > 0:
            lines = lines[:sample]

        for line in tqdm(lines):
            data = build_dataset_for_tags_prediction_task(line, self.node_index_manager, VIS)
            if data is not None and data.tags.sum().item() != 0:
                new_data = []
                if not only_fake:
                    new_data.append(data)

                for _ in range(location_augmentation_factor):
                    add_if_not_none(new_data, replace_location_augmentation(self.node_index_manager, data))
                for _ in range(molecule_similier_factor):
                    add_if_not_none(new_data,
                                    replace_entity_augmentation(self.node_index_manager, data, NodeTypes.molecule,
                                                                "similar"))
                for _ in range(molecule_random_factor):
                    add_if_not_none(new_data,
                                    replace_entity_augmentation(self.node_index_manager, data, NodeTypes.molecule,
                                                                "random"))
                for _ in range(protein_similier_factor):
                    add_if_not_none(new_data,
                                    replace_entity_augmentation(self.node_index_manager, data, NodeTypes.protein,
                                                                "similar"))
                for _ in range(protein_random_factor):
                    add_if_not_none(new_data,
                                    replace_entity_augmentation(self.node_index_manager, data, NodeTypes.protein,
                                                                "random"))
                for _ in range(replace_entity_location):
                    add_if_not_none(new_data, replace_entity_location_augmentation(self.node_index_manager, data))
                if len(new_data) == 0:
                    continue
                if one_per_sample:
                    import random
                    self.reactions.append(random.choice(new_data))
                else:
                    self.reactions.extend(new_data)

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        return self.reactions[idx]


if __name__ == "__main__":
    dataset = ReactionDataset(root="data/items", sample=1, location_augmentation_factor=1,
                              molecule_similier_factor=1, molecule_random_factor=1, protein_similier_factor=1,
                              protein_random_factor=1, replace_entity_location=1)
    # dataset = ReactionDataset(root="data/items")
    print(len(dataset))
    for data in dataset:
        print(data.augmentation_type)
        # print(data)
        # dict_data = data.to_dict()
        # print(dict_data)
        # data2 = HeteroData.from_dict(dict_data)
        # print(data2)
