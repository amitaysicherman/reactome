from common.path_manager import data_path
import pickle
from os.path import join as pjoin
from preprocessing.seq_to_vec import Seq2Vec
from common.args_manager import get_args
from common.data_types import PROTEIN
from tqdm import tqdm
import numpy as np
import os

def read_data(part, task_name):
    seq_file = pjoin(data_path, "CAFA3", f"{part}_seq_{task_name}")
    with open(seq_file, "rb") as f:
        seq = pickle.load(f)
    label_file = pjoin(data_path, "CAFA3", f"{part}_label_{task_name}")
    with open(label_file, "rb") as f:
        label = pickle.load(f)
    return seq, label


if __name__ == "__main__":

    args = get_args()
    self_token = args.self_token
    protein_emd = args.protein_emd
    seq2vec = Seq2Vec(self_token, protein_name=protein_emd)

    output_dir = pjoin(data_path, "CAFA3", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)
    for part in ["train", "test"]:
        for task_name in ["mf", "bp", "cc"]:
            seqs, labels = read_data(part, task_name)
            proteins = []
            for seq in tqdm(seqs):
                proteins.append(seq2vec.to_vec(seq['seq'], PROTEIN))
            proteins = np.array(proteins)
            np.save(pjoin(output_dir, f"{part}_protein_{task_name}_{protein_emd}.npy"), proteins)
            labels = [" ".join([str(i) for i in l]) for l in labels]
            with open(pjoin(output_dir, f"{part}_label_{task_name}_{protein_emd}.txt"), "w") as f:
                f.write("\n".join(labels))
