import dataclasses
from typing import List, Dict
from common.data_types import PROTEIN, DNA, MOLECULE, TEXT
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiModalSeq(nn.Module):
    def __init__(self, size, type_to_vec_dim: Dict[str, int], use_trans):
        super(MultiModalSeq, self).__init__()
        self.d_types = [PROTEIN, MOLECULE, TEXT]
        if size == 's':
            emd_dim = 64
            num_layers = 2
        elif size == 'm':
            emd_dim = 128
            num_layers = 3
        elif size == 'l':
            emd_dim = 256
            num_layers = 4
        else:
            raise ValueError(f"Invalid size: {size}")
        self.t = nn.ModuleDict({k: nn.Linear(type_to_vec_dim[k], emd_dim) for k in self.d_types})
        self.last_lin = nn.Linear(emd_dim, 1)
        self.use_trans = use_trans
        if use_trans:
            encoder_layer = nn.TransformerEncoderLayer(d_model=emd_dim, nhead=2, dim_feedforward=emd_dim * 2,
                                                       batch_first=True)
            self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, batch_data: Dict[str, torch.Tensor], batch_mask: Dict[str, torch.Tensor], return_emd=False):
        all_transformed_data = []
        all_masks = []

        for dtype in self.d_types:
            transformed_data = self.t[dtype](batch_data[dtype])
            mask = batch_mask[dtype].unsqueeze(-1)  # make sure mask is of shape [batch_size, seq_length, 1]
            all_transformed_data.append(transformed_data)
            all_masks.append(mask)

        # Concatenate all transformed data and masks
        concatenated_data = torch.cat(all_transformed_data, dim=1)
        concatenated_mask = torch.cat(all_masks, dim=1)

        if self.use_trans:
            concatenated_data = self.trans(concatenated_data, src_key_padding_mask=concatenated_mask.squeeze(-1) == 0)
        if return_emd:
            return concatenated_data
        # Apply mask
        masked_data = concatenated_data * concatenated_mask
        sum_masked_data = masked_data.sum(dim=1)
        count_masked_data = concatenated_mask.sum(dim=1)

        mean_masked_data = sum_masked_data / torch.clamp(count_masked_data, min=1.0)

        output = self.last_lin(mean_masked_data)
        return output


class EmbModel(nn.Module):
    def __init__(self, n, output_dim):
        super(EmbModel, self).__init__()
        self.emd = nn.Embedding(n, output_dim)

    def forward(self, x, _=""):
        return self.emd(x)


@dataclasses.dataclass
class MultiModalLinearConfig:
    embedding_dim: List[int]
    n_layers: int
    names: List[str]
    hidden_dim: int
    output_dim: List[int]
    dropout: float
    normalize_last: int

    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            for k, v in dataclasses.asdict(self).items():
                if isinstance(v, list) or isinstance(v, tuple):
                    if isinstance(v[0], tuple):
                        v = ["_".join([str(x) for x in y]) for y in v]
                    v = ",".join([str(x) for x in v])
                f.write(f"{k}={v}\n")

    @staticmethod
    def load_from_file(file_name):
        with open(file_name) as f:
            data = {}
            for line in f:
                k, v = line.strip().split("=")
                if k == "names":
                    v = [tuple(v_.split("_")) for v_ in v.split(",")]
                data[k] = v
        return MultiModalLinearConfig(embedding_dim=[int(x) for x in data["embedding_dim"].split(",")],
                                      n_layers=int(data["n_layers"]),
                                      names=data["names"],
                                      hidden_dim=int(data["hidden_dim"]),
                                      output_dim=[int(x) for x in data["output_dim"].split(",")],
                                      dropout=float(data["dropout"]),
                                      normalize_last=int(data["normalize_last"]))


class MiltyModalLinear(nn.Module):
    def __init__(self, config: MultiModalLinearConfig):
        super(MiltyModalLinear, self).__init__()
        self.names = ["_".join(x) if isinstance(x, tuple) else x for x in config.names]
        self.normalize_last = config.normalize_last
        self.dropout = nn.Dropout(config.dropout)
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.layers = nn.ModuleList()
        embedding_dim = {k: v for k, v in zip(self.names, config.embedding_dim)}
        output_dim = {k: v for k, v in zip(self.names, config.output_dim)}
        if config.n_layers == 1:
            self.layers.append(nn.ModuleDict({k: nn.Linear(v, output_dim[k]) for k, v in embedding_dim.items()}))
        else:
            self.layers.append(nn.ModuleDict({k: nn.Linear(v, config.hidden_dim) for k, v in embedding_dim.items()}))
            for _ in range(config.n_layers - 2):
                self.layers.append(
                    nn.ModuleDict(
                        {k: nn.Linear(config.hidden_dim, config.hidden_dim) for k, v in embedding_dim.items()}))
            self.layers.append(
                nn.ModuleDict({k: nn.Linear(config.hidden_dim, output_dim[k]) for k, v in embedding_dim.items()}))

    def have_type(self, type_):
        if isinstance(type_, tuple):
            type_ = "_".join(type_)
        return type_ in self.names

    def forward(self, x, type_):
        if isinstance(type_, tuple):
            type_ = "_".join(type_)
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).float().to(device)
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        for layer in self.layers[:-1]:
            x = F.relu(layer[type_](x))
            x = self.dropout(x)
        x = self.layers[-1][type_](x)
        if self.normalize_last:
            return F.normalize(x, dim=-1)
        else:
            return x


def concat_all_to_one_typs(model: MiltyModalLinear, x, src_type):
    res = []
    for name in model.names:
        src_name, dst_name = name.split("_")
        if src_name == src_type:
            if model.have_type(name):
                res.append(model(x, name))
            else:
                print(f"Warning: model does not have type {name}")
                res.append(x)
    return torch.cat(res, dim=-1)


def apply_model(model: MiltyModalLinear, x, type_):
    if type_ == DNA:
        return torch.Tensor(x)
    if "_" in model.names[0]:
        return concat_all_to_one_typs(model, x, type_)
    else:
        if not model.have_type(type_):
            print(f"Warning: model does not have type {type_}")
            return torch.Tensor(x)
        return model(x, type_)
