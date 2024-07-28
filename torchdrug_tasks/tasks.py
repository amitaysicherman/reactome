from dataclasses import dataclass
from torchdrug_tasks.models import DataType, LinFuseModel, PairTransFuseModel,FuseModel
from torchdrug import datasets
from torchdrug import metrics
from torch import nn
import torch


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


name_to_task = {
    "BetaLactamase": Task("BetaLactamase", datasets.BetaLactamase, LinFuseModel, nn.MSELoss, metrics.r2,
                          DataType.PROTEIN, 1),
    "Fluorescence": Task("Fluorescence", datasets.Fluorescence, LinFuseModel, nn.MSELoss, metrics.r2,
                         DataType.PROTEIN, 1),
    "Stability": Task("Stability", datasets.Stability, LinFuseModel, nn.CrossEntropyLoss, metrics.accuracy,
                      DataType.PROTEIN, 2),
    "Solubility": Task("Solubility", datasets.Stability, LinFuseModel, nn.CrossEntropyLoss, metrics.accuracy,
                       DataType.PROTEIN, 2),
    "BinaryLocalization": Task("BinaryLocalization", datasets.BinaryLocalization, LinFuseModel, nn.CrossEntropyLoss,
                               metrics.accuracy, DataType.PROTEIN, 2),
    "SubcellularLocalization": Task("SubcellularLocalization", datasets.SubcellularLocalization, LinFuseModel,
                                    nn.CrossEntropyLoss, metrics.accuracy, DataType.PROTEIN, 10),
    "Fold": Task("Fold", datasets.Fold, LinFuseModel, nn.CrossEntropyLoss, metrics.accuracy, DataType.PROTEIN, 1195),
    "SecondaryStructure": Task("SecondaryStructure", datasets.SecondaryStructure, LinFuseModel, nn.CrossEntropyLoss,
                               metrics.accuracy, DataType.PROTEIN, 3),
    "ProteinNet": Task("ProteinNet", datasets.ProteinNet, PairTransFuseModel, nn.BCEWithLogitsLoss, metrics.accuracy,
                       DataType.PROTEIN, 2, DataType.PROTEIN),
    "HumanPPI": Task("HumanPPI", datasets.HumanPPI, PairTransFuseModel, nn.BCEWithLogitsLoss, metrics.accuracy,
                     DataType.PROTEIN, 2, DataType.PROTEIN),
    "YeastPPI": Task("YeastPPI", datasets.YeastPPI, PairTransFuseModel, nn.BCEWithLogitsLoss, metrics.accuracy,
                     DataType.PROTEIN, 2, DataType.PROTEIN),
    "PPIAffinity": Task("PPIAffinity", datasets.PPIAffinity, PairTransFuseModel, nn.MSELoss, metrics.r2,
                        DataType.PROTEIN, 1, DataType.PROTEIN),
    "BindingDB": Task("BindingDB", datasets.BindingDB, PairTransFuseModel, nn.MSELoss, metrics.r2, DataType.PROTEIN, 1,
                      DataType.MOLECULE),
    "PDBBind": Task("PDBBind", datasets.PDBBind, PairTransFuseModel, nn.MSELoss, metrics.r2, DataType.PROTEIN, 1,
                    DataType.MOLECULE),
    "BACE": Task("BACE", datasets.BACE, LinFuseModel, nn.CrossEntropyLoss, metrics.area_under_roc, DataType.MOLECULE, 2),
    "BBBP": Task("BBBP", datasets.BBBP, LinFuseModel, nn.CrossEntropyLoss, metrics.area_under_roc, DataType.MOLECULE, 2),
    "ClinTox": Task("ClinTox", datasets.ClinTox, LinFuseModel, nn.CrossEntropyLoss, metrics.area_under_roc, DataType.MOLECULE,
                    2),
    "HIV": Task("HIV", datasets.HIV, LinFuseModel, nn.CrossEntropyLoss, metrics.area_under_roc, DataType.MOLECULE, 2),
    "SIDER": Task("SIDER", datasets.SIDER, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc, DataType.MOLECULE, 27),
    "Tox21": Task("Tox21", datasets.Tox21, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc, DataType.MOLECULE, 12),
}

