import dataclasses
import random

from dataset.index_manger import NodesIndexManager, COMPLEX_NODE_ID
from model.tagging import tag
from collections import defaultdict
from common.data_types import UNKNOWN_ENTITY_TYPE, NodeTypes, EdgeTypes, REAL, FAKE_LOCATION_ALL, FAKE_PROTEIN, \
    FAKE_MOLECULE
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
from common.utils import reaction_from_str
from common.path_manager import reactions_file

ET = EdgeTypes()
NT = NodeTypes()

REACTION = NT.reaction


def complex_index_to_node_index(complex_id):
    return f'c_{complex_id}'


def reaction_to_nx(reaction, node_index_manager: NodesIndexManager):
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
        node = node_index_manager.name_to_node[element.get_db_identifier()]
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
    return G


def get_reaction_entities(reaction, check_output):
    if check_output:
        return reaction.inputs + reaction.outputs + sum([c.entities for c in reaction.catalysis], [])
    return reaction.inputs + sum([c.entities for c in reaction.catalysis], [])


def have_unkown_nodes(reaction, node_index_manager: NodesIndexManager, check_output=False):
    entitites = get_reaction_entities(reaction, check_output)
    for e in entitites:
        if node_index_manager.name_to_node[e.get_db_identifier()].type == UNKNOWN_ENTITY_TYPE:
            return True
    return False


def have_dna_nodes(reaction, node_index_manager: NodesIndexManager, check_output=False):
    entitites = get_reaction_entities(reaction, check_output)
    for e in entitites:
        if node_index_manager.name_to_node[e.get_db_identifier()].type == NT.dna:
            return True
    return False


def reaction_to_data(reaction: str, node_index_manager: NodesIndexManager, fake_task: bool):
    reaction = reaction_from_str(reaction)
    if have_unkown_nodes(reaction, node_index_manager):
        return None
    g = reaction_to_nx(reaction, node_index_manager)
    bp = node_index_manager.bp_name_to_index[
        reaction.biological_process[0]]  # TODO: handle more then one biological_process.

    if fake_task:
        tags = torch.Tensor([0]).to(torch.float32)
    else:
        tags = tag(reaction)
        tags = torch.Tensor(dataclasses.astuple(tags)).to(torch.float32)
    return nx_to_torch_geometric(g, tags=tags, augmentation_type=REAL, bp=torch.LongTensor([bp]))


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
            node_ids = [COMPLEX_NODE_ID] * len(node_ids)
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
        hetero_graph.edge_index_dict.keys()
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
    new_hetero_data.bp = data.bp.clone()
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


@dataclasses.dataclass
class AugmentationsFactors:
    location_augmentation_factor: int = 0
    molecule_similier_factor: int = 0
    molecule_random_factor: int = 0
    protein_similier_factor: int = 0
    protein_random_factor: int = 0

    def get_name_factors(self):
        return ["location"] * self.location_augmentation_factor + [
            "molecule_similier"] * self.molecule_similier_factor + ["molecule_random"] * self.molecule_random_factor + [
            "protein_similier"] * self.protein_similier_factor + ["protein_random"] * self.protein_random_factor


def get_default_augmentation_factors(data_aug="protein"):
    if data_aug == "protein":
        return AugmentationsFactors(location_augmentation_factor=0, molecule_similier_factor=0,
                                    molecule_random_factor=0,
                                    protein_similier_factor=0, protein_random_factor=1)
    elif data_aug == "molecule":
        return AugmentationsFactors(location_augmentation_factor=0, molecule_similier_factor=0,
                                    molecule_random_factor=1,
                                    protein_similier_factor=0, protein_random_factor=0)
    elif data_aug == "location":
        return AugmentationsFactors(location_augmentation_factor=1, molecule_similier_factor=0,
                                    molecule_random_factor=0,
                                    protein_similier_factor=0, protein_random_factor=0)
    else:
        return AugmentationsFactors(location_augmentation_factor=2, molecule_random_factor=1, protein_random_factor=1)


def apply_augmentation(data, node_index_manager: NodesIndexManager, augmentation_type: str):
    if augmentation_type == "location":
        return replace_location_augmentation(node_index_manager, data)
    if augmentation_type == "molecule_similier":
        return replace_entity_augmentation(node_index_manager, data, NodeTypes.molecule, "similar")
    if augmentation_type == "molecule_random":
        return replace_entity_augmentation(node_index_manager, data, NodeTypes.molecule, "random")
    if augmentation_type == "protein_similier":
        return replace_entity_augmentation(node_index_manager, data, NodeTypes.protein, "similar")
    if augmentation_type == "protein_random":
        return replace_entity_augmentation(node_index_manager, data, NodeTypes.protein, "random")
    return None


class ReactionDataset:

    def __init__(self, node_index_manager: NodesIndexManager, lines, augmentations_factors: AugmentationsFactors,
                 only_fake=False, one_per_sample=False, fake_task=True):
        self.node_index_manager = node_index_manager
        self.reactions = []

        for line in tqdm(lines):
            data = reaction_to_data(line, self.node_index_manager, fake_task)
            if data is not None and (fake_task or data.tags.sum().item() != 0):
                new_data = []
                if not only_fake:
                    new_data.append(data)
                for name in augmentations_factors.get_name_factors():
                    add_if_not_none(new_data, apply_augmentation(data, self.node_index_manager, name))
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


def get_data(node_index_manager: NodesIndexManager, augmentations_factors: AugmentationsFactors = None, sample=0,
             fake_task=True, data_aug="protein"):
    if augmentations_factors is None:
        augmentations_factors = get_default_augmentation_factors(data_aug)
    train_lines, val_lines, test_lines = get_reactions(sample)
    train_dataset = ReactionDataset(node_index_manager, train_lines, augmentations_factors,
                                    fake_task=fake_task).reactions
    val_dataset = ReactionDataset(node_index_manager, val_lines, augmentations_factors, fake_task=fake_task).reactions
    test_dataset = ReactionDataset(node_index_manager, test_lines, augmentations_factors, fake_task=fake_task).reactions

    print(f"Train dataset size: {len(train_dataset)}", f"Val dataset size: {len(val_dataset)}",
          f"Test dataset size: {len(test_dataset)}")

    tags = torch.stack([reaction.tags for reaction in train_dataset])
    pos_classes_weights = (1 - tags.mean(dim=0)) / tags.mean(dim=0)

    return train_dataset, val_dataset, test_dataset, pos_classes_weights


def get_reactions(sample_count=0):
    with open(reactions_file) as f:
        lines = f.readlines()
    lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
    train_val_index = int(0.7 * len(lines))
    val_test_index = int(0.85 * len(lines))
    train_lines = lines[:train_val_index]
    val_lines = lines[train_val_index:val_test_index]
    test_lines = lines[val_test_index:]
    if sample_count > 0:
        train_lines = train_lines[:sample_count]
        val_lines = val_lines[:sample_count]
        test_lines = test_lines[:sample_count]
    return train_lines, val_lines, test_lines


if __name__ == "__main__":
    node_index_manager = NodesIndexManager()
    train_dataset, val_dataset, test_dataset, pos_classes_weights = get_data(node_index_manager, None, sample=10,
                                                                             fake_task=True)
    for data in test_dataset:
        print(data.edge_index_dict.keys())
        edge_index_dict = {key: data.edge_index_dict[key] for key in data.edge_index_dict.keys()}
