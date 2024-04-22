import dataclasses

import index_manger
from index_manger import NodesIndexManager, EdgeTypes, NodeTypes, node_colors
from tagging import tag
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
from biopax_parser import reaction_from_str

ET = EdgeTypes()
NT = NodeTypes()

REACTION = NT.reaction
VIS = False


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


def build_dataset_for_tags_prediction_task(reaction: str, node_index_manager: NodesIndexManager, vis: bool = False):
    reaction = reaction_from_str(reaction)
    g, tags = get_nx_for_tags_prediction_task(reaction, node_index_manager)
    if vis:
        title = f"{reaction.name}\n{tags}"
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        plot_graph(g, node_index_manager, ax, title)
        plt.show()
    return nx_to_torch_geometric(g, tags=torch.Tensor(dataclasses.astuple(tags)).to(torch.float32))


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
        hetero_graph[*edge_type].edge_index = edges
    for k, v in kwargs.items():
        hetero_graph[k] = v
    hetero_graph.validate(raise_on_error=True)
    return hetero_graph


class ReactionDataset:
    def __init__(self, root="data/items", sample=0):
        self.root = root
        self.node_index_manager = NodesIndexManager(root)
        self.reactions = []
        with open(f'{root}/reaction.txt') as f:
            lines = f.readlines()
        import random
        random.seed(42)
        random.shuffle(lines)
        if sample > 0:
            lines = lines[:sample]
        for line in tqdm(lines):
            self.reactions.append(build_dataset_for_tags_prediction_task(line, self.node_index_manager, VIS))

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        return self.reactions[idx]


if __name__ == "__main__":
    dataset = ReactionDataset(root="data/items", sample=5)
    for data in dataset:
        # print(data)
        pass
