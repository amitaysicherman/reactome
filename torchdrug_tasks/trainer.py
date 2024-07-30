import dataclasses
import os
import torch
from common.data_types import Config
from common.path_manager import scores_path
from torchdrug_tasks.dataset import get_dataloaders
from torchdrug_tasks.tasks import name_to_task, Task
from torchdrug_tasks.models import LinFuseModel, PairTransFuseModel
from torchdrug import metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def metric_prep_predictions(preds, metric):
    if metric.__name__ == "area_under_roc" or metric.__name__ == "area_under_prc":
        return torch.sigmoid(preds).flatten()
    elif metric.__name__ == "accuracy" or metric.__name__ == "f1_max":
        if preds.shape[1] == 1:
            probs_class_1 = torch.sigmoid(preds)
            probs_class_0 = 1 - probs_class_1
            preds = torch.cat((probs_class_0, probs_class_1), dim=1)
            print(preds.shape)
            print(preds)
        return preds
    else:
        return preds.flatten()


class Scores:

    def __init__(self, preds=None, reals=None):
        self.auc: float = 0
        self.auprc: float = 0
        self.acc: float = 0
        self.f1_max: float = 0
        if preds is not None:
            self.calcualte(preds, reals)

    def calcualte(self, preds, reals):
        auc_pred = metric_prep_predictions(preds, metrics.area_under_roc)
        auprc_pred = metric_prep_predictions(preds, metrics.area_under_prc)
        acc_pred = metric_prep_predictions(preds, metrics.accuracy)
        f1_max_pred = metric_prep_predictions(preds, metrics.f1_max)

        self.auc = metrics.area_under_roc(auc_pred, reals.flatten()).item()
        self.auprc = metrics.area_under_prc(auprc_pred, reals.flatten()).item()
        self.acc = metrics.accuracy(acc_pred, reals.flatten()).item()
        self.f1_max = metrics.f1_max(f1_max_pred, reals).item()

    def __repr__(self):
        return f"AUC: {self.auc}, AUPRC: {self.auprc}, ACC: {self.acc}, F1: {self.f1_max}\n"

    def get_metrics(self):
        return [self.auc, self.auprc, self.acc, self.f1_max]

    def get_metrics_names(self):
        return ["auc", "auprc", "acc", "f1_max"]


class ScoresManager:
    def __init__(self):
        self.valid_scores = Scores()
        self.test_scores = Scores()

    def update(self, valid_score: Scores, test_score: Scores):
        improved = False
        if valid_score.auc > self.valid_scores.auc:
            self.valid_scores.auc = valid_score.auc
            self.test_scores.auc = test_score.auc
            improved = True
        if valid_score.auprc > self.valid_scores.auprc:
            self.valid_scores.auprc = valid_score.auprc
            self.test_scores.auprc = test_score.auprc
            improved = True
        if valid_score.acc > self.valid_scores.acc:
            self.valid_scores.acc = valid_score.acc
            self.test_scores.acc = test_score.acc
            improved = True
        if valid_score.f1_max > self.valid_scores.f1_max:
            self.valid_scores.f1_max = valid_score.f1_max
            self.test_scores.f1_max = test_score.f1_max
            improved = True
        return improved


def run_epoch(model, loader, optimizer, criterion, metric, part):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    for *all_x, labels in loader:
        if len(all_x) == 1:
            x = all_x[0]
            x = x.float().to(device)
            output = model(x)

        else:
            x1, x2 = all_x
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            output = model(x1, x2)

        optimizer.zero_grad()
        labels = labels.float().to(device)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            labels = labels.squeeze(1).long()

        loss = criterion(output, labels)
        if part == "train":
            loss.backward()
            optimizer.step()
        reals.append(labels)
        preds.append(output)
    if part != "train":
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        return Scores(preds, reals)
    else:
        return None


def get_model_from_task(task: Task, dataset, conf, fuse_base, drop_out, n_layers, hidden_dim, fuse_model=None):
    model_class = task.model
    input_dim_1 = dataset.x1.shape[1]
    dtype_1 = task.dtype1
    if task.dtype2 is not None:
        input_dim_2 = dataset.x2.shape[1]
        dtype_2 = task.dtype2
    else:
        input_dim_2 = None
        dtype_2 = None
    output_dim = task.output_dim
    if task.model == LinFuseModel:
        return model_class(input_dim=input_dim_1, input_type=dtype_1, output_dim=output_dim, conf=conf,
                           fuse_base=fuse_base, fuse_model=fuse_model, drop_out=drop_out, n_layers=n_layers,
                           hidden_dim=hidden_dim)
    elif task.model == PairTransFuseModel:
        return model_class(input_dim_1=input_dim_1, dtpye_1=dtype_1, input_dim_2=input_dim_2, dtype_2=dtype_2,
                           output_dim=output_dim, conf=conf, fuse_base=fuse_base, fuse_model=fuse_model,
                           drop_out=drop_out, n_layers=n_layers, hidden_dim=hidden_dim)
    else:
        raise ValueError("Unknown model")


def train_model_with_config(config: dict, task_name: str, fuse_base: str, mol_emd: str, protein_emd: str,
                            print_output=False, max_no_improve=15, fuse_model=None):
    use_fuse = config["use_fuse"]
    use_model = config["use_model"]
    bs = config["bs"]
    lr = config["lr"]

    drop_out = config["drop_out"]
    n_layers = config["n_layers"]
    hidden_dim = config["hidden_dim"]

    task = name_to_task[task_name]
    train_loader, valid_loader, test_loader = get_dataloaders(task_name, mol_emd, protein_emd, bs)
    metric = task.metric
    if task.criterion == torch.nn.CrossEntropyLoss:
        train_labels = train_loader.dataset.labels
        positive_sample_weight = train_labels.sum() / len(train_labels)
        negative_sample_weight = 1 - positive_sample_weight
        pos_weight = negative_sample_weight / positive_sample_weight
        criterion = task.criterion(pos_weight=pos_weight)
    else:
        criterion = task.criterion()
    if use_fuse and use_model:
        conf = Config.both
    elif use_fuse:
        conf = Config.our
    elif use_model:
        conf = Config.PRE
    else:
        if print_output:
            print("No model selected")
        return -1e6, -1e6

    model = get_model_from_task(task, train_loader.dataset, conf, fuse_base=fuse_base, drop_out=drop_out,
                                n_layers=n_layers, hidden_dim=hidden_dim, fuse_model=fuse_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if print_output:
        print(model)
    no_improve = 0
    scores_manager = ScoresManager()
    # best_valid_score = -1e6
    # best_test_score = -1e6
    for epoch in range(250):
        _ = run_epoch(model, train_loader, optimizer, criterion, metric, "train")
        with torch.no_grad():
            val_score = run_epoch(model, valid_loader, optimizer, criterion, metric, "val")
            test_score = run_epoch(model, test_loader, optimizer, criterion, metric, "test")

        if print_output:
            print(epoch, val_score, test_score)
        improved = scores_manager.update(val_score, test_score)
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    if print_output:
        print("Best Test scores\n", scores_manager.test_scores)
        output_file = f"{scores_path}/torchdrug.csv"

        if not os.path.exists(output_file):
            names = ["task_name", "mol_emd", "protein_emd", "conf"] + scores_manager.test_scores.get_metrics_names()
            with open(output_file, "w") as f:
                f.write(",".join(names) + "\n")
        values = [task_name, mol_emd, protein_emd, conf.value] + scores_manager.test_scores.get_metrics()
        with open(output_file, "a") as f:
            f.write(",".join(map(str, values)) + "\n")
        return scores_manager.test_scores.get_metrics()


def main(args, fuse_model=None):
    config = {
        "use_fuse": args.cafa_use_fuse,
        "use_model": args.cafa_use_model,
        "bs": args.dp_bs,
        "lr": args.dp_lr,
        'n_layers': args.dp_n_layers,
        'hidden_dim': args.dp_hidden_dim,
        'drop_out': args.dp_drop_out
    }
    train_model_with_config(config, args.task_name, args.dp_fuse_base, args.mol_emd, args.protein_emd, args.dp_print,
                            args.max_no_improve, fuse_model=fuse_model)


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
