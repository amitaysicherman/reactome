from preprocessing.seq_to_vec import Seq2Vec
from common.path_manager import data_path
import os
from tqdm import tqdm
from common.data_types import PROTEIN, MOLECULE
import numpy as np
from common.args_manager import get_args

if __name__ == "__main__":
    args = get_args()
    self_token = args.self_token
    protein_emd = args.protein_emd
    mol_emd = args.mol_emd

    data_type = args.prep_reactome_dtype
    seq2vec = Seq2Vec(self_token, protein_name=protein_emd, mol_name=mol_emd)

    dataset = args.db_dataset

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
        if data_type in [PROTEIN, 'all']:
            proteins.append(protein_name)
            proteins_vec.append(seq2vec.to_vec(fasta, PROTEIN))
        if data_type in [MOLECULE, 'all']:
            molecules.append(mol_name)
            molecules_vec.append(seq2vec.to_vec(smiles, MOLECULE))
        if data_type == "all":
            labels.append(label)

    if len(molecules):
        with open(os.path.join(base_dir, f"{dataset}_{mol_emd}_molecules.txt"), "w") as f:
            f.write("\n".join(molecules))
        np.save(os.path.join(base_dir, f"{dataset}_{mol_emd}_molecules.npy"), np.array(molecules_vec))
    if len(proteins):
        with open(os.path.join(base_dir, f"{dataset}_{protein_emd}_proteins.txt"), "w") as f:
            f.write("\n".join(proteins))
        np.save(os.path.join(base_dir, f"{dataset}_{protein_emd}_proteins.npy"), np.array(proteins_vec))
    if len(labels):
        with open(os.path.join(base_dir, f"{dataset}_labels.txt"), "w") as f:
            f.write("\n".join(labels))
