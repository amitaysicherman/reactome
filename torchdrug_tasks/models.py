import torch
from enum import Enum
import os

from common.data_types import Config
from model.models import MultiModalLinearConfig, MiltyModalLinear
from common.path_manager import model_path


class DataType(Enum):
    MOLECULE = 'molecule_protein'
    PROTEIN = 'protein_protein'


def load_fuse_model(base_dir):
    base_dir = str(os.path.join(model_path, base_dir))
    cp_names = os.listdir(base_dir)
    cp_name = [x for x in cp_names if x.endswith(".pt")][0]
    print(f"Load model {base_dir}/{cp_name}")
    cp_data = torch.load(f"{base_dir}/{cp_name}", map_location=torch.device('cpu'))
    config_file = os.path.join(base_dir, 'config.txt')
    config = MultiModalLinearConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = MiltyModalLinear(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


def get_layers(dims):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
    return layers


class FuseModel(torch.nn.Module):
    def __init__(self, conf: Config, fuse_model=None, fuse_base=""):
        super().__init__()
        if conf == Config.both:
            self.use_fuse = True
            self.use_model = True
        elif conf == Config.PRE:
            self.use_fuse = True
            self.use_model = False
        else:
            self.use_fuse = False
            self.use_model = True

        if self.use_fuse:
            if fuse_model is None:
                self.fuse_model, dim = load_fuse_model(fuse_base)
            else:
                self.fuse_model = fuse_model
                dim = fuse_model.output_dim[0]
            self.fuse_dim = dim


class LinFuseModel(FuseModel):
    def __init__(self, input_dim: int, input_type: DataType, output_dim: int, conf: Config,fuse_model=None, fuse_base=""):
        super().__init__(conf, fuse_model, fuse_base)
        self.input_dim = 0
        if self.use_fuse:
            self.input_dim += self.fuse_dim
        if self.use_model:
            self.input_dim += input_dim
        self.dtype = input_type
        self.layers = get_layers([self.input_dim] + [512] + [output_dim])

    def forward(self, data):
        x = []
        if self.use_fuse:
            x.append(self.fuse_model(data, self.dtype).detach())
        if self.use_model:
            x.append(data)
        x = torch.concat(x, dim=1)
        return self.layers(x)


class PairTransFuseModel(FuseModel):
    def __init__(self, input_dim_1: int, dtpye_1: DataType, input_dim_2: int, dtype_2: DataType, output_dim: int,
                 conf: Config,
                 trans_dim=256, n_layers=2, nhead=2, dropout=0.5, fuse_model=None,
                 fuse_base="", **kwargs):
        super().__init__(conf, fuse_model, fuse_base)

        if self.use_fuse:
            self.x1_fuse_linear = torch.nn.Linear(self.fuse_dim, trans_dim)
            self.x2_fuse_linear = torch.nn.Linear(self.fuse_dim, trans_dim)
        if self.use_model:
            self.x1_model_linear = torch.nn.Linear(input_dim_1, trans_dim)
            self.x2_model_linear = torch.nn.Linear(input_dim_2, trans_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=trans_dim, nhead=nhead, dim_feedforward=trans_dim * 2,
                                                         batch_first=True, dropout=dropout)
        self.trans = torch.nn.Sequential(
            torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers),
            torch.nn.Linear(trans_dim, output_dim)
        )
        self.x1_type = dtpye_1
        self.x2_type = dtype_2

    def forward(self, x1, x2):
        x = []
        if self.use_fuse:
            x1_fuse = self.fuse_model(x1, self.x1_type).detach()
            x.append(self.x1_fuse_linear(x1_fuse))
            x2_fuse = self.fuse_model(x2, self.x2_type).detach()
            x.append(self.x2_fuse_linear(x2_fuse))
        if self.use_model:
            x1_model = self.x1_model_linear(x1)
            x.append(x1_model)
            x2_model = self.x2_model_linear(x2)
            x.append(x2_model)
        x = torch.stack(x, dim=1)
        return self.trans(x).mean(dim=1)
