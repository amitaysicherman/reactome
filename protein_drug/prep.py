from preprocessing.seq_to_vec import Seq2Vec
from common.path_manager import data_path
import os
from tqdm import tqdm
from common.data_types import PROTEIN, MOLECULE
import numpy as np
from common.args_manager import get_args

if __name__ == "__main__":

    args = get_args()
    dataset = args.db_dataset

    self_token = args.self_token
    protein_emd = args.protein_emd
    data_type = args.prep_reactome_dtype
    name = args.prep_reactome_name
    seq2vec = Seq2Vec(self_token, protein_emd)

    base_dir = os.path.join(data_path, "protein_drug")
    input_file = os.path.join(base_dir, f"{dataset}.txt")
    with open(input_file) as f:
        lines = f.read().splitlines()
    molecules = []
    molecules_vec = []

    proteins = []
    proteins_vec = []

    labels = []

    for line in tqdm(lines):
        if dataset == "human":
            smiles, fasta, label = line.split(" ")
            mol_name = smiles
            protein_name = fasta
        else:
            mol_name, protein_name, smiles, fasta, label = line.split(" ")
        molecules.append(mol_name)
        proteins.append(protein_name)
        labels.append(label)
        molecules_vec.append(seq2vec.to_vec(smiles, MOLECULE))
        proteins_vec.append(seq2vec.to_vec(fasta, PROTEIN))

    with open(os.path.join(base_dir, f"{dataset}_{name}_molecules.txt"), "w") as f:
        f.write("\n".join(molecules))
    with open(os.path.join(base_dir, f"{dataset}_{name}_proteins.txt"), "w") as f:
        f.write("\n".join(proteins))
    with open(os.path.join(base_dir, f"{dataset}_{name}_labels.txt"), "w") as f:
        f.write("\n".join(labels))

    np.save(os.path.join(base_dir, f"{dataset}_{name}_molecules.npy"), np.array(molecules_vec))
    np.save(os.path.join(base_dir, f"{dataset}_{name}_proteins.npy"), np.array(proteins_vec))
