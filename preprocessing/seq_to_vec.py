import os
import re
from abc import ABC

import numpy as np
import torch
from npy_append_array import NpyAppendArray
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoModelForMaskedLM, AutoTokenizer, BertConfig

from common.utils import TYPE_TO_VEC_DIM
from common.data_types import DNA, PROTEIN, MOLECULE, TEXT, EMBEDDING_DATA_TYPES

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 1024


def clip_to_max_len(x: torch.Tensor, max_len: int = MAX_LEN):
    if x.shape[1] <= max_len:
        return x
    last_token = x[:, -1:]
    clipped_x = x[:, :max_len - 1]
    result = torch.cat([clipped_x, last_token], dim=1)
    return result


class ABCSeq2Vec(ABC):
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def to_vec(self, seq: str):
        inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
        inputs = clip_to_max_len(inputs)
        with torch.no_grad():
            hidden_states = self.model(inputs)[0]
        vec = torch.mean(hidden_states[0], dim=0)
        return self.post_process(vec)

    def post_process(self, vec):
        vec = vec.detach().cpu().numpy().flatten()
        return vec.reshape(1, -1)


class Prot2vec(ABCSeq2Vec):
    def __init__(self):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").eval().to(device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)

    def to_vec(self, seq: str):
        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
        ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        input_ids = clip_to_max_len(input_ids)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        attention_mask = clip_to_max_len(attention_mask)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        vec = embedding_repr.last_hidden_state[0].mean(dim=0)
        return self.post_process(vec)


class DNA2Vec(ABCSeq2Vec):
    def __init__(self):
        super().__init__()

        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.model = AutoModel.from_config(config).eval().to(device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)


class Mol2Vec(ABCSeq2Vec):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").base_model.eval().to(
            device)
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")


class BioText2Vec(ABCSeq2Vec):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")
        self.model = AutoModel.from_pretrained("gsarti/biobert-nli").eval().to(device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)


class Seq2Vec:
    def __init__(self):
        self.prot2vec = Prot2vec()
        self.dna2vec = DNA2Vec()
        self.mol2vec = Mol2Vec()
        self.text2vec = BioText2Vec()

    def to_vec(self, seq: str, seq_type: str):
        zeros = np.zeros((1, TYPE_TO_VEC_DIM[seq_type]))
        if seq_type == PROTEIN:
            vec = self.prot2vec.to_vec(seq)
        elif seq_type == DNA:
            vec = self.dna2vec.to_vec(seq)
        elif seq_type == MOLECULE:
            vec = self.mol2vec.to_vec(seq)
        elif seq_type == TEXT:
            vec = self.text2vec.to_vec(seq)
        else:
            print(f"Unknown sequence type: {seq_type}")
            return zeros
        if vec is None:
            return zeros
        return vec


def read_seq_write_vec(seq2vec, input_file_name, output_file_name, seq_type):
    with open(input_file_name) as f:
        seqs = f.read().splitlines()
    missing_count = 0
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    for seq in tqdm(seqs):
        vec = seq2vec.to_vec(seq, seq_type)
        if vec is None:
            missing_count += 1
        with NpyAppendArray(output_file_name) as f:
            f.append(vec)
    print(f"Missing {missing_count}({missing_count / len(seqs):%}) sequences for {seq_type}")


if __name__ == "__main__":
    from common.path_manager import item_path

    seq2vec = Seq2Vec()
    for dt in EMBEDDING_DATA_TYPES:
        read_seq_write_vec(seq2vec, f'{item_path}/{dt}_sequences.txt', f'{item_path}/{dt}_vec.npy', dt)
