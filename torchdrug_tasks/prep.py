# sbatch --time=1-0 --array=1-21 --gres=gpu:A40:1 --mem=64G -c 4 --requeue --wrap="python3 GO/prep.py --task_index $SLURM_ARRAY_TASK_ID-1"
from preprocessing.seq_to_vec import Seq2Vec
import numpy as np
from common.path_manager import data_path
from os.path import join as pjoin
import os
from tqdm import tqdm
from torchdrug import datasets
from torchdrug.data import ordered_scaffold_split
from torchdrug_tasks.tasks import name_to_task, Task
from torchdrug_tasks.models import DataType
from common.data_types import MOLECULE, PROTEIN

base_dir = f"{data_path}/torchdrug/"
os.makedirs(base_dir, exist_ok=True)
SIDER_LABELS = ['Hepatobiliary disorders',
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


def get_vec(seq2vec, x, dtype):
    try:
        if dtype == DataType.MOLECULE:
            return seq2vec.to_vec(x.to_smiles(), MOLECULE)
        elif task.dtype1 == DataType.PROTEIN:
            return seq2vec.to_vec(x.to_sequence().replace(".G", ""), PROTEIN)
        else:
            raise Exception("dtype", dtype)
    except Exception as e:
        print(e)
        return None


def prep_dataset(task: Task, seq2vec, protein_emd, mol_emd):
    output_file = pjoin(base_dir, f"{task.name}_{protein_emd}_{mol_emd}.npz")
    if os.path.exists(output_file):
        return
    dataset = task.dataset(pjoin(base_dir, task.name))
    labels_keys = getattr(task.dataset, 'target_fields')
    if task.name == "SIDER":
        labels_keys = SIDER_LABELS
    if hasattr(task.dataset, "splits"):
        splits = dataset.split()
        if len(splits) == 3:
            train, valid, test = splits
        elif len(splits) > 3:
            train, valid, test, *unused_test = splits
        else:
            raise Exception("splits", getattr(task.dataset, "splits"))

    else:
        train, valid, test = ordered_scaffold_split(dataset, None)
    x1_all = dict()
    x2_all = dict()
    labels_all = dict()
    for split, name in zip([train, valid, test], ["train", "valid", "test"]):
        x1_vecs = []
        x2_vecs = []
        labels = []
        for i in tqdm(range(len(split))):
            key1 = "graph" if task.dtype2 is None else "graph1"
            new_vec = get_vec(seq2vec, split[i][key1], task.dtype1)
            if new_vec is None:
                continue
            if task.dtype2 is not None:
                new_vec_2 = get_vec(seq2vec, split[i]["graph2"], task.dtype2)
                if new_vec_2 is None:
                    continue
                x2_vecs.append(new_vec_2)
            x1_vecs.append(new_vec)

            label = [split[i][key] for key in labels_keys]
            labels.append(label)
        x2_vecs = np.array(x2_vecs)

        x1_all[f'x1_{name}'] = np.array(x1_vecs)[:, 0, :]
        if len(x2_vecs):
            x2_all[f'x2_{name}'] = np.array(x2_vecs)[:, 0, :]
        labels_all[f'label_{name}'] = np.array(labels)
    if len(x2_all):
        np.savez(output_file, **x1_all, **x2_all, **labels_all)
    else:
        np.savez(output_file, **x1_all, **labels_all)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--protein_emd", type=str, default="ESM2")
    parser.add_argument("--mol_emd", type=str, default="pebchem10m")
    parser.add_argument("--self_token", type=str, default="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")
    parser.add_argument("--task_index", type=int, default=-1)

    args = parser.parse_args()
    seq2vec = Seq2Vec(args.self_token, protein_name=args.protein_emd, mol_name=args.mol_emd)

    if args.task_index >= 0:
        names = sorted(list(name_to_task.keys()))
        tasks = [name_to_task[names[args.task_index]]]
    elif args.tasks == "all":
        tasks = [name_to_task.values()]
    else:
        tasks = [name_to_task[args.task]]
    for task in tasks:
        prep_dataset(task, seq2vec, args.protein_emd, args.mol_emd)
