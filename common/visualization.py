import networkx as nx

from common.utils import node_colors
from dataset.dataset_builder import REACTION, NT


def plot_graph(G, ax, title=""):
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
