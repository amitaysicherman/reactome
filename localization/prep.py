import os
from os.path import join as pjoin

import numpy as np

from common.path_manager import data_path
from common.data_types import PROTEIN
import pickle
import lmdb
from common.args_manager import get_args
from preprocessing.seq_to_vec import Seq2Vec
import os
import struct


def compute_md5(file_name, chunk_size=65536):
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def download(url, path, save_file=None, md5=None):
    from six.moves.urllib.request import urlretrieve

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        urlretrieve(url, save_file)
    return save_file


def extract(zip_file, member=None):
    """
    Extract files from a zip file. Currently, ``zip``, ``gz``, ``tar.gz``, ``tar`` file types are supported.

    Parameters:
        zip_file (str): file name
        member (str, optional): extract specific member from the zip file.
            If not specified, extract all members.
    """
    import gzip
    import shutil
    import zipfile
    import tarfile

    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]
    save_path = os.path.dirname(zip_file)

    if extension == ".gz":
        member = os.path.basename(zip_name)
        members = [member]
        save_files = [os.path.join(save_path, member)]
        for _member, save_file in zip(members, save_files):
            with open(zip_file, "rb") as fin:
                fin.seek(-4, 2)
                file_size = struct.unpack("<I", fin.read())[0]
            with gzip.open(zip_file, "rb") as fin:
                if not os.path.exists(save_file) or file_size != os.path.getsize(save_file):
                    with open(save_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
    elif extension in [".tar.gz", ".tgz", ".tar"]:
        tar = tarfile.open(zip_file, "r")
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
        else:
            members = tar.getnames()
            save_files = [os.path.join(save_path, _member) for _member in members]
        for _member, save_file in zip(members, save_files):
            if tar.getmember(_member).isdir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or tar.getmember(_member).size != os.path.getsize(save_file):
                with tar.extractfile(_member) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    elif extension == ".zip":
        zipped = zipfile.ZipFile(zip_file)
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
        else:
            members = zipped.namelist()
            save_files = [os.path.join(save_path, _member) for _member in members]
        for _member, save_file in zip(members, save_files):
            if zipped.getinfo(_member).is_dir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or zipped.getinfo(_member).file_size != os.path.getsize(save_file):
                with zipped.open(_member, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    else:
        raise ValueError("Unknown file extension `%s`" % extension)

    if len(save_files) == 1:
        return save_files[0]
    else:
        return save_path


args = get_args()
protein_emd = args.protein_emd
self_token = args.self_token
seq_to_vec = Seq2Vec(self_token=self_token, protein_name=protein_emd)
is_bin = args.loc_bin
if is_bin:
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz"
    md5 = "5d2309bf1c0c2aed450102578e434f4e"
    dir = "subcellular_localization_2"

else:
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz"
    md5 = "37cb6138b8d4603512530458b7c8a77d"
    dir = "subcellular_localization"

loc_dir = "localization"

target_fields = ["localization"]
path = pjoin(data_path, loc_dir)

if not os.path.exists(pjoin(data_path, loc_dir, dir, f'{dir}_train.lmdb')):
    zip_file = download(url, path, md5=md5)
    extract(zip_file)
    os.remove(zip_file)

sequence_field = "primary"
splits = ["train", "valid", "test"]

sequences = []
locations = []

for split in splits:
    input_file = pjoin(data_path, loc_dir, dir, f'{dir}_{split}.lmdb')
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
