import dataclasses
from typing import List

from torch import nn as nn
from torch.nn import functional as F


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
                if isinstance(v, list):
                    v = ",".join([str(x) for x in v])
                f.write(f"{k}={v}\n")

    @staticmethod
    def load_from_file(file_name):
        with open(file_name) as f:
            data = {}
            for line in f:
                k, v = line.strip().split("=")
                data[k] = v
        return MultiModalLinearConfig(embedding_dim=[int(x) for x in data["embedding_dim"].split(",")],
                                      n_layers=int(data["n_layers"]),
                                      names=data["names"].split(","),
                                      hidden_dim=int(data["hidden_dim"]),
                                      output_dim=[int(x) for x in data["output_dim"].split(",")],
                                      dropout=float(data["dropout"]),
                                      normalize_last=int(data["normalize_last"]))


class MiltyModalLinear(nn.Module):
    def __init__(self, config: MultiModalLinearConfig):
        super(MiltyModalLinear, self).__init__()

        self.normalize_last = config.normalize_last
        self.dropout = nn.Dropout(config.dropout)
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.layers = nn.ModuleList()
        embedding_dim = {k: v for k, v in zip(config.names, config.embedding_dim)}
        output_dim = {k: v for k, v in zip(config.names, config.output_dim)}
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

    def forward(self, x, type_):
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        for layer in self.layers[:-1]:
            x = F.relu(layer[type_](x))
        x = self.layers[-1][type_](x)
        if self.normalize_last:
            return F.normalize(x, dim=-1)
        else:
            return x


