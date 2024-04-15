from nodes_indexes import NodesIndexManager, REACTION_NODE_ID, EdgeTypes, NodeTypes, \
    get_entity_reaction_type, get_location_to_entity, node_colors, bind_edge
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
from biopax_parser import reaction_from_dict
import numpy as np

ET = EdgeTypes()
NT = NodeTypes()

REACTION = NT.reaction
VIS = False


def add_entity(G, entity_list, node_index_manager: NodesIndexManager, is_input: bool, mem_map: dict,
               is_cat=False, disconnect_reaction=False):
    complex_nodes = defaultdict(list)
    for entity in entity_list:
        id = entity.get_unique_id()
        n1 = len(G.nodes)

        node_type = node_index_manager.get_type(id)
        edge_type = get_entity_reaction_type(node_type, is_input, is_cat)
        if not edge_type or not node_type:
            continue
        G.add_node(n1, id=node_index_manager.get_index(id), type=node_type)
        complex_nodes[entity.complex_id].append((n1, node_type))
        if not disconnect_reaction:
            if is_input:
                G.add_edge(n1, REACTION_NODE_ID, type=edge_type)
            else:
                G.add_edge(REACTION_NODE_ID, n1, type=edge_type)

        if entity.location in mem_map:
            n2 = mem_map[entity.location]
        else:
            n2 = len(G.nodes)
            mem_map[entity.location] = n2
            G.add_node(n2, id=node_index_manager.get_index(entity.location), type=NT.location)
            G.add_edge(n2, n2, type=ET.location_self_loop)

        G.add_edge(n2, n1, type=get_location_to_entity(node_type))

        if entity.modifications:
            for modification in entity.modifications:
                modification_index = node_index_manager.get_index(modification)
                if modification_index in mem_map:
                    n3 = mem_map[modification_index]
                else:

                    n3 = len(G.nodes)
                    mem_map[modification_index] = n3
                    G.add_node(n3, id=modification_index, type=NT.text)
                    G.add_edge(n3, n3, type=ET.modification_self_loop)
                assert node_index_manager.get_type(modification) == NT.text
                if node_type != NT.protein:
                    print(node_type, modification)
                    continue
                G.add_edge(n3, n1, type=ET.modification_to_protein)
        if not disconnect_reaction:
            for complex_id, nodes_types in complex_nodes.items():
                if complex_id == 0:
                    continue
                if len(nodes_types) > 1:
                    for i in range(len(nodes_types)):
                        for j in range(i + 1, len(nodes_types)):
                            bind_type = bind_edge(nodes_types[i][1], nodes_types[j][1])
                            G.add_edge(nodes_types[i][0], nodes_types[j][0], type=bind_type)


def get_nx(reaction, node_index_manager: NodesIndexManager):
    G = nx.DiGraph()
    mem_map = dict()
    G.add_node(0, id=REACTION_NODE_ID, type=REACTION)
    add_entity(G, reaction.inputs, node_index_manager, is_input=True, mem_map=mem_map)
    add_entity(G, reaction.outputs, node_index_manager, is_input=False, mem_map=mem_map)
    if reaction.catalysis:
        for c in reaction.catalysis:
            add_entity(G, [e for e in c.entities], node_index_manager, True, mem_map, is_cat=True)
            activity_index = node_index_manager.get_index(c.activity)
            if activity_index in mem_map:
                n = mem_map[activity_index]
            else:
                n = len(G.nodes)
                mem_map[activity_index] = n
                G.add_node(n, id=node_index_manager.get_index(c.activity), type=NT.text)
                G.add_edge(n, n, type=ET.catalysis_activity_self_loop)
            G.add_edge(n, REACTION_NODE_ID, type=ET.catalysis_activity_to_reaction)
    return G


def get_nx_for_output_prediction_task(reaction, node_index_manager: NodesIndexManager):
    G = nx.DiGraph()
    mem_map = dict()
    G.add_node(0, id=REACTION_NODE_ID, type=REACTION)
    add_entity(G, reaction.inputs, node_index_manager, is_input=True, mem_map=mem_map)
    if reaction.catalysis:
        for c in reaction.catalysis:
            add_entity(G, [e for e in c.entities], node_index_manager, True, mem_map, is_cat=True)
            activity_index = node_index_manager.get_index(c.activity)
            if activity_index in mem_map:
                n = mem_map[activity_index]
            else:
                n = len(G.nodes)
                mem_map[activity_index] = n
                G.add_node(n, id=node_index_manager.get_index(c.activity), type=NT.text)
                G.add_edge(n, n, type=ET.catalysis_activity_self_loop)
            G.add_edge(n, REACTION_NODE_ID, type=ET.catalysis_activity_to_reaction)

    real_vectors_indexes = [node_index_manager.get_index(e.get_unique_id()) for e in reaction.outputs]
    fake_indexes = sum([node_index_manager.sample_closest_n_from_k(e.get_unique_id()) for e in reaction.outputs], [])
    fake_indexes = list(set(fake_indexes))
    fake_indexes = [x for x in fake_indexes if x not in real_vectors_indexes]
    output_vectors = [node_index_manager.vector[i] for i in real_vectors_indexes + fake_indexes]
    output_vectors = np.stack(output_vectors)
    output_labels = [1] * len(real_vectors_indexes) + [0] * len(fake_indexes)
    output_labels = torch.tensor(output_labels, dtype=torch.float32)
    return G, output_vectors, output_labels


def plot_graph(G, ax, title=""):
    pos = nx.spring_layout(G)

    node_types = nx.get_node_attributes(G, "type")

    nx.draw_networkx_nodes(G, pos, node_size=150, ax=ax,
                           node_color=[node_colors[node_type] for node_type in node_types.values()])
    nx.draw_networkx_edges(G, pos, width=1, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
    ax.set_title(title)
    ax.axis("off")


def nx_to_torch_geometric(G: nx.Graph, **kwargs):
    nodes = G.nodes(data=True)
    edges = G.edges(data=True)
    hetero_graph = HeteroData()
    nodes_dict = defaultdict(list)
    edges_dict = defaultdict(list)
    nodes_index_to_hetro = dict()
    for i, d in nodes:
        nodes_index_to_hetro[i] = len(nodes_dict[d["type"]])
        nodes_dict[d["type"]].append(d["id"])
    for node_type, node_ids in nodes_dict.items():
        hetero_graph[node_type].x = torch.LongTensor(node_ids).reshape(-1, 1)
    for src, dst, d in edges:
        edges_dict[d["type"]].append([nodes_index_to_hetro[src], nodes_index_to_hetro[dst]])

    for edge_type, edge_list in edges_dict.items():
        edges = torch.IntTensor(edge_list).t().to(torch.int64)
        hetero_graph[*edge_type].edge_index = edges
    for k, v in kwargs.items():
        hetero_graph[k] = v
    hetero_graph.validate(raise_on_error=True)
    return hetero_graph


def reaction_str_to_graph(reaction: str, node_index_manager: NodesIndexManager, vis=False):
    reaction_dict = eval(reaction)
    reaction = reaction_from_dict(reaction_dict)
    g = get_nx(reaction, node_index_manager)
    if vis:
        fig, ax = plt.subplots(figsize=(15, 15))
        plot_graph(g, ax, reaction.name)
        plt.show()
    return nx_to_torch_geometric(g)


def build_dataset_for_node_output_task(reaction: str, node_index_manager: NodesIndexManager):
    reaction_dict = eval(reaction)
    reaction = reaction_from_dict(reaction_dict)
    g, output_vec, y = get_nx_for_output_prediction_task(reaction, node_index_manager)
    return nx_to_torch_geometric(g, output_vec=output_vec, y=y)


class ReactionDataset:
    def __init__(self, root="data/items", sample=0, task=None):
        self.root = root
        self.node_index_manager = NodesIndexManager(root)
        self.reactions = []
        with open(f'{root}/reaction.txt') as f:
            lines = f.readlines()
        if sample > 0:
            lines = lines[:sample]
        for line in tqdm(lines):
            if task == "output_node":
                self.reactions.append(build_dataset_for_node_output_task(line, self.node_index_manager))
            else:
                self.reactions.append(reaction_str_to_graph(line, self.node_index_manager, VIS))

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        return self.reactions[idx]


if __name__ == "__main__":
    dataset = ReactionDataset(sample=10, task="output_node")
    for data in dataset:
        print(data)
