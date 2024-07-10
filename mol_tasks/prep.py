from torchdrug import datasets
from preprocessing.seq_to_vec import Seq2Vec
from common.data_types import MOLECULE
import numpy as np
from os.path import join as pjoin
import os
from common.utils import sent_to_key
from tqdm import tqdm

base_dir = "data/mol/"

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
    mols = []
    labels = []
    mol_output_file = pjoin(base_dir, f"{task_name}_{mol_emd}_molecules.npy")
    labels_output_file = pjoin(base_dir, f"{task_name}_{sent_to_key(label_key)}_label.npy")
    skip_mol = False
    if os.path.exists(mol_output_file):
        skip_mol = True
    for i in tqdm(range(len(dataset))):
        labels.append(dataset[i][label_key])
        if skip_mol:
            continue
        x = dataset[i]['graph'].to_smiles()
        mols.append(seq2vec.to_vec(x, MOLECULE))
    labels = np.array(labels)
    np.save(labels_output_file, labels)
    if skip_mol:
        return
    mols = np.array(mols)
    np.save(mol_output_file, mols)


# BACE:    Binary binding results for a set of inhibitors of human :math:`\beta`-secretase 1(BACE-1).
# BBBP:    Blood-brain barrier penetration.
# ClinTox: Qualitative data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons.
# HIV:     Experimentally measured abilities to inhibit HIV replication.
# SIDER:   Marketed drugs and adverse drug reactions (ADR) dataset, grouped into 27 system organ classes.


tasks = ["BACE", "BBBP", "ClinTox", "HIV", "SIDER"]

task_to_label_keys = {
    "BACE": ["Class"],
    "BBBP": ["p_np"],
    "ClinTox": ["FDA_APPROVED", "CT_TOX"],
    "HIV": ["HIV_active"],
    "SIDER": ['Hepatobiliary disorders',
              'Metabolism and nutrition disorders',
              'Product issues',
              'Eye disorders',
              'Investigations',
              'Musculoskeletal and connective tissue disorders',
              'Gastrointestinal disorders',
              'Social circumstances',
              'Immune system disorders',
              'Reproductive system and breast disorders',
              'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
              'General disorders and administration site conditions',
              'Endocrine disorders',
              'Surgical and medical procedures',
              'Vascular disorders',
              'Blood and lymphatic system disorders',
              'Skin and subcutaneous tissue disorders',
              'Congenital, familial and genetic disorders',
              'Infections and infestations',
              'Respiratory, thoracic and mediastinal disorders',
              'Psychiatric disorders',
              'Renal and urinary disorders',
              'Pregnancy, puerperium and perinatal conditions',
              'Ear and labyrinth disorders',
              'Cardiac disorders',
              'Nervous system disorders',
              'Injury, poisoning and procedural complications']
}
log_file = "log.txt"
for task in tasks:
    for label_key in task_to_label_keys[task]:
        with open(log_file, "a") as f:
            f.write(f"Processing {task} {label_key}\n")
        prep_dataset_part(task, label_key)
