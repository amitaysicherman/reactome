# Description: This file contains common variables and functions used by other files in the project.
import datetime

from matplotlib import pyplot as plt

from common.data_types import CatalystOBJ, Entity, Reaction, UNKNOWN_ENTITY_TYPE, DNA, PROTEIN, MOLECULE, TEXT, \
    NodeTypes, EdgeTypes

TYPE_TO_VEC_DIM = {
    PROTEIN: 1024,
    DNA: 768,
    MOLECULE: 768,
    TEXT: 768
}


def db_to_type(db_name):
    db_name = db_name.lower()
    if db_name == "ensembl":
        return DNA
    elif db_name == "embl":
        return PROTEIN
    elif db_name == "uniprot" or db_name == "uniprot isoform":
        return PROTEIN
    elif db_name == "chebi":
        return MOLECULE
    elif db_name == "guide to pharmacology":
        return MOLECULE
    elif db_name == "go":
        return TEXT
    elif db_name == "text":
        return TEXT
    elif db_name == "ncbi nucleotide":
        return DNA
    elif db_name == "pubchem compound":
        return MOLECULE
    else:
        return UNKNOWN_ENTITY_TYPE
        # raise ValueError(f"Unknown database name: {db_name}")


def catalyst_from_dict(d: dict) -> CatalystOBJ:
    entities = [Entity(**e) for e in d["entities"]]
    activity = d["activity"]
    return CatalystOBJ(entities, activity)


def reaction_from_dict(d: dict) -> Reaction:
    name = d["name"]
    inputs = [Entity(**e) for e in d["inputs"]]
    outputs = [Entity(**e) for e in d["outputs"]]
    catalysis = [catalyst_from_dict(c) for c in d["catalysis"]]
    year, month, day = d["date"].split("_")
    date = datetime.date(int(year), int(month), int(day))
    reactome_id = d["reactome_id"]
    return Reaction(name, inputs, outputs, catalysis, date, reactome_id)


def reaction_from_str(s: str) -> Reaction:
    return reaction_from_dict(eval(s))


def get_node_types():
    return [NodeTypes.reaction, NodeTypes.complex, NodeTypes.location, NodeTypes.protein, NodeTypes.dna,
            NodeTypes.molecule, NodeTypes.text]


color_palette = plt.get_cmap("tab10")
node_colors = {node_type: color_palette(i) for i, node_type in enumerate(get_node_types())}


def get_edges_values():
    attributes = dir(EdgeTypes)
    edges = []
    for attr in attributes:
        value = getattr(EdgeTypes, attr)
        if isinstance(value, tuple) and len(value) == 3:
            edges.append(value)
    return edges


# def load_model(learned_embedding_dim=128, hidden_channels=128, num_layers=3, root="../data/items",
#                layer_type="SAGEConv", return_reaction_embedding=False, model_path="../data/model/model.pt",out_channels=1,fuse=False):
#     nodes_index_manager = NodesIndexManager(root,fuse_vec=fuse)
#
#     model = HeteroGNN(nodes_index_manager, hidden_channels=hidden_channels, out_channels=out_channels,
#                       num_layers=num_layers,
#                       learned_embedding_dim=learned_embedding_dim, train_all_emd=False, save_activation=True,
#                       conv_type=layer_type, return_reaction_embedding=return_reaction_embedding)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model, nodes_index_manager


