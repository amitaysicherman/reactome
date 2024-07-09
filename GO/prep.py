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
protein_emd = args.protein_emd
self_token = args.self_token
output_protein = pjoin(data_path, "GO", f"{protein_emd}.npy")
if os.path.exists(output_protein):
    print(f"Skip {output_protein}")
    exit(0)

cc_dataset = datasets.GeneOntology("data/GO/", branch="CC", transform=ProteinView(view="residue"),
                                atom_feature=None, bond_feature=None)
bp_dataset = datasets.GeneOntology("data/GO/", branch="BP", transform=ProteinView(view="residue"),
                                atom_feature=None, bond_feature=None)
mf_dataset = datasets.GeneOntology("data/GO/", branch="MF", transform=ProteinView(view="residue"),
                                atom_feature=None, bond_feature=None)

seq2vec = Seq2Vec(use_cache=True, protein_name=protein_emd, self_token=self_token)
proteins = []
cc_labels = []
bp_labels = []
mf_labels = []
for i in range(len(cc_dataset)):
    cc_seq = cc_dataset[i]['graph'].to_sequence().replace(".G", "")
    bp_seq = bp_dataset[i]['graph'].to_sequence().replace(".G", "")
    mf_seq = mf_dataset[i]['graph'].to_sequence().replace(".G", "")
    assert cc_seq == bp_seq == mf_seq
    proteins.append(seq2vec.to_vec(cc_seq, PROTEIN))
    cc_labels.append(cc_dataset[i]['targets'].numpy())
    bp_labels.append(bp_dataset[i]['targets'].numpy())
    mf_labels.append(mf_dataset[i]['targets'].numpy())

proteins = np.array(proteins)
cc_labels = np.array(cc_labels)
bp_labels = np.array(bp_labels)
mf_labels = np.array(mf_labels)

output_protein = pjoin(data_path, "GO", f"{protein_emd}.npy")
output_cc_label = pjoin(data_path, "GO", f"CC_label.npy")
output_bp_label = pjoin(data_path, "GO", f"BP_label.npy")
output_mf_label = pjoin(data_path, "GO", f"MF_label.npy")

np.save(output_protein, proteins)
np.save(output_cc_label, cc_labels)
np.save(output_bp_label, bp_labels)
np.save(output_mf_label, mf_labels)
