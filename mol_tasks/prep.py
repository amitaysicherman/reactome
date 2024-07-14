from torchdrug import datasets
from torchdrug.data import ordered_scaffold_split

from preprocessing.seq_to_vec import Seq2Vec
from common.data_types import MOLECULE
import numpy as np
from common.path_manager import data_path
from os.path import join as pjoin
import os
from common.utils import sent_to_key
from tqdm import tqdm
from common.data_types import mol_task_to_label_keys, mol_tasks

base_dir = f"{data_path}/mol/"

from common.args_manager import get_args

args = get_args()
self_token = args.self_token
protein_emd = args.protein_emd
mol_emd = args.mol_emd
seq2vec = Seq2Vec(self_token, protein_name=protein_emd, mol_name=mol_emd)

name_to_dataset = {
    "BACE": datasets.BACE,
    "BBBP": datasets.BBBP,
    "ClinTox": datasets.ClinTox,
    "HIV": datasets.HIV,
    "SIDER": datasets.SIDER
}


def prep_dataset_part(task_name, label_key):
    dataset = name_to_dataset[task_name](pjoin(base_dir, task_name))
    mols_dict = dict()
    labels_dict = dict()
    mol_output_file = pjoin(base_dir, f"{task_name}_{mol_emd}_molecules.npy")
    labels_output_file = pjoin(base_dir, f"{task_name}_label.npy")
    train, valid, test = ordered_scaffold_split(dataset, None)
    for split, name in zip([train, valid, test], ["train", "valid", "test"]):
        mols=[]
        labels=[]
        for i in tqdm(range(len(split))):
            try:
                x = split[i]['graph'].to_smiles()
                mols.append(seq2vec.to_vec(x, MOLECULE))
                if type(label_key) == list:
                    label = [split[i][key] for key in label_key]
                    labels.append(label)
                else:
                    labels.append([split[i][label_key]])
            except:
                print(f"Error processing {i}")

    labels = np.array(labels)
    np.save(labels_output_file, labels)
    mols = np.array(mols)
    np.save(mol_output_file, mols)


# BACE:    Binary binding results for a set of inhibitors of human :math:`\beta`-secretase 1(BACE-1).
# BBBP:    Blood-brain barrier penetration.
# ClinTox: Qualitative data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons.
# HIV:     Experimentally measured abilities to inhibit HIV replication.
# SIDER:   Marketed drugs and adverse drug reactions (ADR) dataset, grouped into 27 system organ classes.


log_file = "log.txt"
for task in mol_tasks:
    label_key = mol_task_to_label_keys[task]
    with open(log_file, "a") as f:
        f.write(f"Processing {task} {label_key}\n")
    prep_dataset_part(task, label_key)
