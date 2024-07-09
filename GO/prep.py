import torch
import numpy as np
from os.path import join as pjoin
from torchdrug import data, datasets
from torchdrug.transforms import ProteinView
from preprocessing.seq_to_vec import Seq2Vec
from common.args_manager import get_args
from common.data_types import PROTEIN
from common.path_manager import data_path
from tqdm import tqdm
import os

args = get_args()
task = args.go_task
protein_emd = args.protein_emd
self_token = args.self_token
output_protein = pjoin(data_path, "GO", f"{task}_{protein_emd}.npy")
if os.path.exists(output_protein):
    print(f"Skip {output_protein}")
    exit(0)

dataset = datasets.GeneOntology("data/GO/", branch=task, transform=ProteinView(view="residue"),
                                atom_feature=None, bond_feature=None)
seq2vec = Seq2Vec(use_cache=True, protein_name=protein_emd, self_token=self_token)
proteins = []
labels = []
for data in tqdm(dataset):
    seq = data['graph'].to_sequence().replace(".G", "")
    proteins.append(seq2vec.to_vec(seq, PROTEIN))
    labels.append(data['targets'].numpy())

proteins = np.array(proteins)
labels = np.array(labels)

output_protein = pjoin(data_path, "GO", f"{task}_{protein_emd}.npy")
output_label = pjoin("data", f"{task}_label.npy")
np.save(output_protein, proteins)
np.save(output_label, labels)
