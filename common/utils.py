# Description: This file contains common variables and functions used by other files in the project.
import datetime
import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from common.data_types import CatalystOBJ, Entity, Reaction, UNKNOWN_ENTITY_TYPE, DNA, PROTEIN, MOLECULE, TEXT, \
    NodeTypes
from common.path_manager import model_path, scores_path
from model.models import MultiModalLinearConfig, MiltyModalLinear, EmbModel

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
    biological_process = d["biological_process"].split("_")
    return Reaction(name, inputs, outputs, catalysis, date, reactome_id, biological_process)


def reaction_from_str(s: str) -> Reaction:
    return reaction_from_dict(eval(s))


def get_node_types():
    return [NodeTypes.reaction, NodeTypes.complex, NodeTypes.location, NodeTypes.protein, NodeTypes.dna,
            NodeTypes.molecule, NodeTypes.text]


color_palette = plt.get_cmap("tab10")
node_colors = {node_type: color_palette(i) for i, node_type in enumerate(get_node_types())}


def load_fuse_model(name, fuse_pretrained_start):
    model_cp = glob.glob(f'{model_path}/fuse_{name}/fuse_*.pt')
    if len(model_cp) == 0:
        return None
    if len(model_cp) > 1:
        print(f"More than one model found for {name}")
    model_cp = model_cp[0]
    cp_data = torch.load(model_cp, map_location=torch.device('cpu'))

    if fuse_pretrained_start:
        config_file = f'{model_path}/fuse_{name}/config.txt'
        config = MultiModalLinearConfig.load_from_file(config_file)
        model = MiltyModalLinear(config)
    else:
        x, y = cp_data['emd.weight'].shape
        model = EmbModel(x, y)

    model.load_state_dict(cp_data)
    model.eval()
    return model


# def get_last_epoch_model(model_dir, cp_idx):
#     if cp_idx != -1:
#         return f"{model_dir}/model_{cp_idx}.pt"
#     files = os.listdir(f'{model_dir}')
#     ephocs = [int(x.split("_")[-1].replace(".pt", "")) for x in files if x.startswith("model")]
#     last_epoch = max(ephocs)
#     return f"{model_dir}/model_{last_epoch}.pt"


def get_best_fuse_cp(score_file):
    with open(score_file, "r") as f:
        lines = f.read().splitlines()

    lines = [x for x in lines if "Test" in x]
    best_score = 0
    best_index = len(lines) - 1
    for index, line in enumerate(lines):
        score = float(line.split("AUC")[1].split("(")[0])
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def get_best_gnn_cp(name, aug_data):
    score_file = f'{scores_path}/gnn_{name}.txt'
    with open(score_file, "r") as f:
        lines = f.read().splitlines()
    best_score = 0
    best_index = - 1
    for i in range(len(lines) // 4):
        line = lines[i * 4 + 3]
        line = eval(line.replace("nan", "0"))
        score = line[f"test/fake_{aug_data}"]
        if score > best_score:
            best_score = score
            best_index = i
    best_cp = f"{model_path}/gnn_{name}/model_{best_index}.pt"
    return best_cp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prepare_files(run_name, skip_if_exists=False):
    save_dir = f"{model_path}/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        if skip_if_exists:
            print(f"Skip {run_name}")
            exit(0)
    for file_name in os.listdir(save_dir):
        if file_name.endswith(".pt"):
            os.remove(f"{save_dir}/{file_name}")
    scores_file = f"{scores_path}/fuse_{run_name}.txt"
    if os.path.exists(scores_file):
        os.remove(scores_file)
    return save_dir, scores_file
