import os
from os.path import join as pjoin

import numpy as np

from common.path_manager import data_path
from common.data_types import PROTEIN
from torchdrug import data, utils
import pickle
import lmdb
from common.args_manager import get_args
from preprocessing.seq_to_vec import Seq2Vec

args = get_args()
protein_emd = args.protein_emd
self_token = args.self_token
seq_to_vec = Seq2Vec(protein_emd, self_token)
is_bin = args.loc_bin
if is_bin:
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz"
    md5 = "5d2309bf1c0c2aed450102578e434f4e"
    dir = "subcellular_localization"

else:
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz"
    md5 = "37cb6138b8d4603512530458b7c8a77d"
    dir = "subcellular_localization_2"

loc_dir = "localization"

target_fields = ["localization"]
path = pjoin(data_path, loc_dir)

zip_file = utils.download(url, path, md5=md5)
utils.extract(zip_file)
os.remove(zip_file)

sequence_field = "primary"
splits = ["train", "valid", "test"]

sequences = []
locations = []

for split in splits:
    input_file = pjoin(data, loc_dir, dir, f'{dir}_{split}.lmdb')
    env = lmdb.open(input_file, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_sample = pickle.loads(txn.get("num_examples".encode()))
        for i in range(num_sample):
            item = pickle.loads(txn.get(str(i).encode()))
            sequences.append(seq_to_vec.to_vec(item[sequence_field], PROTEIN))
            locations.append(item["localization"])

output_protein = pjoin(data_path, loc_dir, f"{dir}_{protein_emd}.npy")
output_target = pjoin(data_path, loc_dir, f"{dir}_target.npy")
np.save(output_protein, np.array(sequences))
np.save(output_target, np.array(locations))
