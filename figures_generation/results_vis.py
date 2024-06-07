# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from dataset.index_manger import NodesIndexManager
# from common.data_types import Reaction, NodeTypes
# import networkx as nx
# from model.tagging import tag
# import torch
# from model.tagging import ReactionTag
# import random
# from dataset.dataset_builder import reaction_to_nx, nx_to_torch_geometric, have_unkown_nodes, \
#     replace_location_augmentation, replace_entity_augmentation
# import dataclasses
# import numpy as np
# import seaborn as sns
# import requests
# from typing import List
# from common.utils import load_model, reaction_from_str, node_colors
# from common.path_manager import model_path, item_path, reactions_file
#
# sns.set_theme(style="white")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# NT = NodeTypes()
# FAKE_TASK = False
#
#
# def get_go(id):
#     response = requests.get(f"https://api.geneontology.org/api/ontology/term/{id}")
#     if response.status_code == 200:
#         return response.json().get("label", "")
#     else:
#         return id
#
#
# REACTION_INDEX = 0
#
# node_colors[NT.molecule] = node_colors[NT.protein]
# node_colors[NT.dna] = node_colors[NT.protein]
#
#
# def add_nodes(g, nodes):
#     g.add_node(0, type=NT.reaction, name="Reaction")
#
#     for node in nodes:
#         node_index = len(g)
#         g.add_node(node_index, type=NT.protein, name=node.name.replace(" ", "\n"))
#         if node.complex_id != 0:
#             g.add_node(node.complex_id + 100, type=NT.complex, name="Complex")
#             g.add_edge(node.complex_id + 100, REACTION_INDEX)
#             g.add_edge(node_index, node.complex_id + 100)
#         else:
#             g.add_edge(node_index, REACTION_INDEX)
#         location_index = len(g)
#         g.add_node(location_index, type=NT.location, name=node.location.replace(" ", "\n"))
#         g.add_edge(location_index, node_index)
#         for mod in node.modifications:
#             mod_index = len(g)
#             g.add_node(mod_index, type=NT.text, name=mod.replace(" ", "\n"))
#             g.add_edge(mod_index, node_index)
#     return g
#
#
# def hetero_data_to_nx(data, nodes_index_manager: NodesIndexManager, name_to_hr: dict):
#     g = nx.DiGraph()
#     g.add_node(0, type=NT.reaction, name="Reaction")
#
#     for etype in data.edge_index_dict:
#         src_type, _, dst_type = etype
#         for src_type_index, dst_type_index in data.edge_index_dict[etype].t().tolist():
#             src = data.x_dict[src_type][src_type_index].item()
#             dst = data.x_dict[dst_type][dst_type_index].item()
#             src_name = nodes_index_manager.index_to_node[src].name
#             dst_name = nodes_index_manager.index_to_node[dst].name
#             src_index = nodes_index_manager.index_to_node[src].index
#             dst_index = nodes_index_manager.index_to_node[dst].index
#
#             if src_type == NT.location:
#                 if dst_type == NT.location:
#                     continue
#                 else:
#                     src_index = -1 * len(g)  # workaround to create new location in the graph.
#             src_name = name_to_hr[src_name.lower()]
#             if src_type == NT.text and src_name.startswith("go"):
#                 src_name = get_go(src_name.upper())
#             src_name = src_name.replace(" ", "\n")
#             dst_name = name_to_hr[dst_name.lower()].replace(" ", "\n")
#
#             g.add_node(dst_index, type=dst_type, name=dst_name)
#             g.add_node(src_index, type=src_type, name=src_name)
#             g.add_edge(src_index, dst_index)
#
#     return g
#
#
# def add_activities(g, reaction):
#     activities = [c.activity for c in reaction.catalysis]
#     for node in activities:
#         node_index = len(g)
#         g.add_node(node_index, type=NT.text, name=get_go(node).replace(" ", "\n"))
#         g.add_edge(node_index, REACTION_INDEX)
#     return g
#
#
# def plot_g(g, ax, title=""):
#     pos = nx.nx_agraph.graphviz_layout(g, prog="dot", args="")
#     colors = [node_colors[g.nodes[node]["type"]] for node in g.nodes]
#     labels = {node: g.nodes[node]["name"].replace(" ", "\n") for node in g.nodes}
#     nx.draw(g.to_undirected(), pos, node_color=colors, with_labels=True, labels=labels, node_size=3000, font_size=8,
#             ax=ax)
#     if title:
#         ax.set_title(title)
#     return ax
#
#
# def plot_probs(probs, ax):
#     tag_names = ReactionTag().get_names()
#     ax.barh(tag_names, probs)
#     ax.set_yticklabels(tag_names, rotation=45, ha='right')
#     ax.set_title("Predicted Labels")
#     return ax
#
#
# def get_name_to_opt_hr(reactions: List[Reaction], index_manager: NodesIndexManager) -> dict:
#     name_to_hr = {}
#     for node in index_manager.index_to_node.values():
#         if node.type == NT.location:
#             name_to_hr[node.name.lower()] = node.name.replace(" ", "\n")
#         elif node.type == NT.text:
#             name_to_hr[node.name.lower()] = node.name.split("@")[1].lower()
#
#     for reaction in reactions:
#         catalisis = sum([c.entities for c in reaction.catalysis], [])
#         for node in reaction.inputs + reaction.outputs + catalisis:
#             index_name = node.get_db_identifier()
#             name_to_hr[index_name.lower()] = node.name
#     name_to_hr['reaction'] = 'Reaction'
#     name_to_hr['complex'] = 'Complex'
#     return name_to_hr
#
#
# def get_fake_data(data, nodes_index_manager: NodesIndexManager):
#     return [
#         ("Change Location", replace_location_augmentation(nodes_index_manager, data)),
#         ("Change Random Protein", replace_entity_augmentation(nodes_index_manager, data, NT.protein, "random")),
#         ("Change Random Molecule", replace_entity_augmentation(nodes_index_manager, data, NT.molecule, "random")),
#         ("Change Similar Protein", replace_entity_augmentation(nodes_index_manager, data, NT.protein, "similar")),
#         ("Change Similar Molecule", replace_entity_augmentation(nodes_index_manager, data, NT.molecule, "similar"))
#     ]
#
#
# with open(reactions_file) as f:
#     lines = f.readlines()
# lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
# reactions = [reaction_from_str(line) for line in tqdm(lines)]
# name_to_hr = get_name_to_opt_hr(reactions, NodesIndexManager(root))
# reactions = reactions[int(len(lines) * 0.8):]
# model, nodes_index_manager = load_model(learned_embedding_dim=256, hidden_channels=256, num_layers=3, out_channels=5,
#                                         model_path="/data/models_checkpoints/model_mlc_256_2.pt")
# for reaction in random.choices(reactions, k=10):
#     if have_unkown_nodes(reaction, nodes_index_manager):
#         continue
#     if len(reaction.inputs) > 5:
#         continue
#     g = reaction_to_nx(reaction, nodes_index_manager)
#     tags = tag(reaction)
#
#     data = nx_to_torch_geometric(g, tags=torch.Tensor(dataclasses.astuple(tags)).to(torch.float32))
#
#     if data is None:
#         continue
#     if data.tags.sum() == 0:
#         continue
#     data = data.to(device)
#     real_tags = ",".join(str(tag(reaction)).split())
#     with torch.no_grad():
#         out = model(data.x_dict, data.edge_index_dict)
#     out_prob = torch.sigmoid(out).detach().cpu().numpy()[0]
#
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1, 2]})
#     plot_probs(out_prob, ax2)
#
#     g_input = nx.DiGraph()
#     catalisis = sum([c.entities for c in reaction.catalysis], [])
#     add_nodes(g_input, reaction.inputs + catalisis)
#     add_activities(g_input, reaction)
#
#     g_output = nx.DiGraph()
#     add_nodes(g_output, reaction.outputs)
#
#     plot_g(g_input, ax1, "Input")
#     plot_g(g_output, ax3, "Output")
#     title = f'{reaction.name} ({reaction.reactome_id})'
#     title += f'({real_tags})'
#     fig.suptitle(title)
#     fig.tight_layout()
#     plt.savefig(f"../data/fig/{reaction.reactome_id}.png", dpi=300)
#     plt.show()
#
#     if not FAKE_TASK:
#         continue
#
#     fake_location_data = replace_location_augmentation(nodes_index_manager, data)
#     with torch.no_grad():
#         out = model(fake_location_data.x_dict, fake_location_data.edge_index_dict)
#     out_prob = torch.sigmoid(out).detach().cpu().numpy()[0]
#
#     for (name, data) in get_fake_data(data, nodes_index_manager):
#         if data is None:
#             continue
#         fake_graph = hetero_data_to_nx(data, nodes_index_manager, name_to_hr)
#         with torch.no_grad():
#             out = model(data.x_dict, data.edge_index_dict)
#         out_prob = np.maximum(out_prob, torch.sigmoid(out).detach().cpu().numpy()[0])
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={'width_ratios': [2, 1]})
#         plot_probs(out_prob, ax2)
#         plot_g(fake_graph, ax1, "")
#         fig.suptitle(name)
#         fig.tight_layout()
#         plt.show()
