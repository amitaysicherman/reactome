from dataclasses import dataclass
from torchdrug_tasks.models import DataType, LinFuseModel, PairTransFuseModel, FuseModel
from torchdrug import datasets
from torch import nn
import torch
import enum


class PrepType(enum.Enum):
    torchdrug = "torchdrug"
    drugtarget = "drugtarget"



@dataclass
class Task:
    name: str
    dataset: object
    model: FuseModel
    criterion: object
    metric: object
    dtype1: DataType
    output_dim: int
    dtype2: DataType = None
    prep_type: PrepType = PrepType.torchdrug


def mse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return -1 * mse

classification="classification"
regression="regression"

name_to_task = {
    "BetaLactamase": Task("BetaLactamase", datasets.BetaLactamase, LinFuseModel, nn.MSELoss, regression,
                          DataType.PROTEIN, 1),
    "Fluorescence": Task("Fluorescence", datasets.Fluorescence, LinFuseModel, nn.MSELoss, regression,
                         DataType.PROTEIN, 1),
    "Stability": Task("Stability", datasets.Stability, LinFuseModel, nn.MSELoss, regression,
                      DataType.PROTEIN, 1),
    "Solubility": Task("Solubility", datasets.Solubility, LinFuseModel, nn.BCEWithLogitsLoss, classification,
                       DataType.PROTEIN, 1),
    "SubcellularLocalization": Task("SubcellularLocalization", datasets.SubcellularLocalization, LinFuseModel,
                                    nn.CrossEntropyLoss, classification, DataType.PROTEIN, 10),
    "HumanPPI": Task("HumanPPI", datasets.HumanPPI, PairTransFuseModel, nn.BCEWithLogitsLoss, classification,
                     DataType.PROTEIN, 1, DataType.PROTEIN),
    "YeastPPI": Task("YeastPPI", datasets.YeastPPI, PairTransFuseModel, nn.BCEWithLogitsLoss, classification,
                     DataType.PROTEIN, 1, DataType.PROTEIN),
    "PPIAffinity": Task("PPIAffinity", datasets.PPIAffinity, PairTransFuseModel, nn.MSELoss, regression,
                        DataType.PROTEIN, 1, DataType.PROTEIN),
    "BindingDB": Task("BindingDB", datasets.BindingDB, PairTransFuseModel, nn.MSELoss, regression, DataType.PROTEIN, 1,
                      DataType.MOLECULE),
    "PDBBind": Task("PDBBind", datasets.PDBBind, PairTransFuseModel, nn.MSELoss, regression, DataType.PROTEIN, 1,
                    DataType.MOLECULE),
    "BACE": Task("BACE", datasets.BACE, LinFuseModel, nn.BCEWithLogitsLoss, classification, DataType.MOLECULE,
                 1),
    "BBBP": Task("BBBP", datasets.BBBP, LinFuseModel, nn.BCEWithLogitsLoss, classification, DataType.MOLECULE,
                 1),
    "ClinTox": Task("ClinTox", datasets.ClinTox, LinFuseModel, nn.BCEWithLogitsLoss, classification,
                    DataType.MOLECULE,
                    2),
    "HIV": Task("HIV", datasets.HIV, LinFuseModel, nn.BCEWithLogitsLoss, classification, DataType.MOLECULE, 1),
    "SIDER": Task("SIDER", datasets.SIDER, LinFuseModel, nn.BCEWithLogitsLoss, classification,
                  DataType.MOLECULE, 27),
    "Tox21": Task("Tox21", datasets.Tox21, LinFuseModel, nn.BCEWithLogitsLoss, classification,
                  DataType.MOLECULE, 12),
    "DrugBank": Task("DrugBank", None, PairTransFuseModel, nn.BCEWithLogitsLoss, classification,
                     DataType.PROTEIN, 1, DataType.MOLECULE, PrepType.drugtarget),
    "Davis": Task("Davis", None, PairTransFuseModel, nn.BCEWithLogitsLoss, classification, DataType.PROTEIN,
                  1, DataType.MOLECULE, PrepType.drugtarget),
    "KIBA": Task("KIBA", None, PairTransFuseModel, nn.BCEWithLogitsLoss, classification, DataType.PROTEIN, 1,
                 DataType.MOLECULE, PrepType.drugtarget),
}
