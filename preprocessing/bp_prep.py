from common.path_manager import item_path
import requests
from tqdm import tqdm
from preprocessing.seq_to_vec import BioText2Vec
from common.utils import TYPE_TO_VEC_DIM, TEXT
import os
from npy_append_array import NpyAppendArray

import numpy as np

bp_path = f"{item_path}/bp.txt"
bp_seq = f"{item_path}/bp_sequences.txt"
if os.path.exists(bp_seq):
    os.remove(bp_seq)
bp_vec = f"{item_path}/bp_vec.npy"
if os.path.exists(bp_vec):
    os.remove(bp_vec)

vec_dim = TYPE_TO_VEC_DIM[TEXT]


def bp_to_text(go_id):
    url = f"https://api.geneontology.org/api/ontology/term/{go_id}"

    response = requests.get(url)
    if response.status_code == 200:
        json_resp = response.json()
    else:
        print(f"Failed to retrieve url : {url} ")
        return ""
    label = json_resp.get("label", "")
    definition = json_resp.get("definition", "")
    return f"{label}. {definition}" if label or definition else ""


def text_to_vec(text, text2vec):
    zeros = np.zeros((1, vec_dim))
    vec = text2vec.to_vec(text)
    if vec is None:
        return zeros
    return vec


with open(bp_path) as f:
    lines = f.read().splitlines()
seqs = []
for line in tqdm(lines):
    line = line.split("@")[0]
    seqs.append(bp_to_text(line))
with open(bp_seq, "w") as f:
    for seq in seqs:
        f.write(seq + "\n")

text2vec = BioText2Vec()
for seq in tqdm(seqs):
    vec = text_to_vec(seq, text2vec)
    with NpyAppendArray(bp_vec) as f:
        f.append(vec)
