from dataclasses import dataclass
from torchdrug_tasks.models import DataType, LinFuseModel, PairTransFuseModel, FuseModel
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

def mse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return -1*mse



name_to_task = {
    "BetaLactamase": Task("BetaLactamase", datasets.BetaLactamase, LinFuseModel, nn.MSELoss, mse_metric,
                          DataType.PROTEIN, 1),
    "Fluorescence": Task("Fluorescence", datasets.Fluorescence, LinFuseModel, nn.MSELoss, mse_metric,
                         DataType.PROTEIN, 1),
    "Stability": Task("Stability", datasets.Stability, LinFuseModel, nn.CrossEntropyLoss, metrics.accuracy,
                      DataType.PROTEIN, 1),
    "Solubility": Task("Solubility", datasets.Stability, LinFuseModel, nn.MSELoss, mse_metric,
                       DataType.PROTEIN, 1),
    # "BinaryLocalization": Task("BinaryLocalization", datasets.BinaryLocalization, LinFuseModel, nn.CrossEntropyLoss,
    #                            metrics.accuracy, DataType.PROTEIN, 2),
    # "SubcellularLocalization": Task("SubcellularLocalization", datasets.SubcellularLocalization, LinFuseModel,
    #                                 nn.CrossEntropyLoss, metrics.accuracy, DataType.PROTEIN, 10),
    # "Fold": Task("Fold", datasets.Fold, LinFuseModel, nn.CrossEntropyLoss, metrics.accuracy, DataType.PROTEIN, 1195),
    "SecondaryStructure": Task("SecondaryStructure", datasets.SecondaryStructure, LinFuseModel, nn.CrossEntropyLoss,
                               metrics.accuracy, DataType.PROTEIN, 3),
    "ProteinNet": Task("ProteinNet", datasets.ProteinNet, PairTransFuseModel, nn.CrossEntropyLoss, metrics.accuracy,
                       DataType.PROTEIN, 2, DataType.PROTEIN),
    "HumanPPI": Task("HumanPPI", datasets.HumanPPI, PairTransFuseModel, nn.CrossEntropyLoss, metrics.accuracy,
                     DataType.PROTEIN, 2, DataType.PROTEIN),
    "YeastPPI": Task("YeastPPI", datasets.YeastPPI, PairTransFuseModel, nn.CrossEntropyLoss, metrics.accuracy,
                     DataType.PROTEIN, 2, DataType.PROTEIN),
    "PPIAffinity": Task("PPIAffinity", datasets.PPIAffinity, PairTransFuseModel, nn.MSELoss, mse_metric,
                        DataType.PROTEIN, 1, DataType.PROTEIN),
    "BindingDB": Task("BindingDB", datasets.BindingDB, PairTransFuseModel, nn.MSELoss, mse_metric, DataType.PROTEIN, 1,
                      DataType.MOLECULE),
    "PDBBind": Task("PDBBind", datasets.PDBBind, PairTransFuseModel, nn.MSELoss, mse_metric, DataType.PROTEIN, 1,
                    DataType.MOLECULE),
    "BACE": Task("BACE", datasets.BACE, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc, DataType.MOLECULE,
                 1),
    "BBBP": Task("BBBP", datasets.BBBP, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc, DataType.MOLECULE,
                 1),
    "ClinTox": Task("ClinTox", datasets.ClinTox, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc,
                    DataType.MOLECULE,
                    2),
    "HIV": Task("HIV", datasets.HIV, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc, DataType.MOLECULE, 1),
    "SIDER": Task("SIDER", datasets.SIDER, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc,
                  DataType.MOLECULE, 27),
    "Tox21": Task("Tox21", datasets.Tox21, LinFuseModel, nn.BCEWithLogitsLoss, metrics.area_under_roc,
                  DataType.MOLECULE, 12),
}
