from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoTokenizer, \
    BertConfig
import re
import torch
from abc import ABC
from npy_append_array import NpyAppendArray
import os
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 1024
VEC_DIM = 1024


class ABCSeq2Vec(ABC):
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def to_vec(self, seq: str):
        inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
        with torch.no_grad():
            hidden_states = self.model(inputs)[0]
        vec = torch.mean(hidden_states[0], dim=0)
        return self.post_process(vec)

    def post_process(self, vec):
        vec = vec.detach().cpu().numpy().flatten()
        if len(vec) < VEC_DIM:
            vec = np.concatenate((vec, np.zeros(VEC_DIM - len(vec))))
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
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
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
        zeros = np.zeros((1, VEC_DIM))
        if seq == "":
            return zeros
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        if seq_type == "Protein":
            return self.prot2vec.to_vec(seq)
        elif seq_type == "DNA":
            return self.dna2vec.to_vec(seq)
        elif seq_type == "Molecule":
            if len(seq) > 512:
                seq = seq[:512]
            return self.mol2vec.to_vec(seq)
        elif seq_type == "Text":
            return self.text2vec.to_vec(seq)
        else:
            print(f"Unknown sequence type: {seq_type}")
            return zeros


seq2vec = Seq2Vec()


def read_seq_write_vec(input_file_name, output_file_name, seq_type="", split_value="\n"):
    with open(input_file_name) as f:
        seqs = f.read().split(split_value)
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    for seq in tqdm(seqs):
        if seq_type == "":
            seq_type, *seq = seq.split(" ")
            seq = " ".join(seq)
        vec = seq2vec.to_vec(seq, seq_type)
        with NpyAppendArray(output_file_name) as f:
            f.append(vec)


BASE_DIR = "./data/items/"
read_seq_write_vec(f'{BASE_DIR}entities_sequences.txt', f'{BASE_DIR}entities_vec.npy')
read_seq_write_vec(f'{BASE_DIR}catalyst_activities_sequences.txt', f'{BASE_DIR}catalyst_activities_vec.npy')
read_seq_write_vec(f'{BASE_DIR}modifications.txt', f'{BASE_DIR}modifications_vec.npy', "Text")
read_seq_write_vec(f'{BASE_DIR}modifications_enrich.txt', f'{BASE_DIR}modifications_enrich_vec.npy', "Text",
                   "\nAMITAY_END\n"
                   )


