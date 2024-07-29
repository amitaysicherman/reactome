import os
import torch
from common.data_types import Config
from common.path_manager import scores_path
from torchdrug_tasks.dataset import get_dataloaders
from torchdrug_tasks.tasks import name_to_task, Task
from torchdrug_tasks.models import LinFuseModel, PairTransFuseModel
from ray import tune

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def metric_prep_predictions(preds, metric):
    if metric.__name__ == "area_under_roc":
        return torch.sigmoid(preds).flatten()
    elif metric.__name__ == "accuracy":
        return preds
    else:
        return preds.flatten()


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
        preds = metric_prep_predictions(preds, metric)
        score = metric(preds, reals.flatten()).item()
        return score
    else:
        return 0


def get_model_from_task(task: Task, dataset, conf, fuse_base, fuse_model=None):
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
                           fuse_base=fuse_base, fuse_model=fuse_model)
    elif task.model == PairTransFuseModel:
        return model_class(input_dim_1=input_dim_1, dtpye_1=dtype_1, input_dim_2=input_dim_2, dtype_2=dtype_2,
                           output_dim=output_dim, conf=conf, fuse_base=fuse_base, fuse_model=fuse_model)
    else:
        raise ValueError("Unknown model")


def train_model_with_config(config: dict, task_name: str, fuse_base: str, mol_emd: str, protein_emd: str,
                            print_output=False, max_no_improve=10, tune_mode=False, fuse_model=None):
    use_fuse = config["use_fuse"]
    use_model = config["use_model"]
    bs = config["bs"]
    lr = config["lr"]
    task = name_to_task[task_name]
    train_loader, valid_loader, test_loader = get_dataloaders(task_name, mol_emd, protein_emd, bs)
    metric = task.metric
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
        if tune_mode:
            tune.report(best_valid_score=-1e6, best_test_score=-1e6)
        return -1e6, -1e6
    model = get_model_from_task(task, train_loader.dataset, conf, fuse_base=fuse_base, fuse_model=fuse_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if print_output:
        print(model)
    no_improve = 0
    best_valid_score = -1e6
    best_test_score = -1e6
    for epoch in range(250):
        train_score = run_epoch(model, train_loader, optimizer, criterion, metric, "train")
        with torch.no_grad():
            val_score = run_epoch(model, valid_loader, optimizer, criterion, metric, "val")
            test_score = run_epoch(model, test_loader, optimizer, criterion, metric, "test")

        if print_output:
            print(epoch, train_score, val_score, test_score)
        if val_score > best_valid_score:
            best_valid_score = val_score
            best_test_score = test_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    if tune_mode:
        tune.report(best_valid_score=best_valid_score, best_test_score=best_test_score)
    else:
        if print_output:
            print("Best Test scores\n", best_test_score)
            task_output_prefix = f"{task_name}_{mol_emd}_{protein_emd}"
            output_file = f"{scores_path}/{task_output_prefix}torchdrug.csv"
            if not os.path.exists(output_file):
                names = "name,mol,prot,conf,prefix,task,bs,lr,score\n"
                with open(output_file, "w") as f:
                    f.write(names)
            with open(output_file, "a") as f:
                f.write(
                    f'{task_name},{mol_emd},{protein_emd},{conf},{task_output_prefix},{task_name},{bs},{lr},{best_test_score}\n')
    return best_valid_score, best_test_score


def main(args, fuse_model=None, tune_mode=False):
    config = {
        "use_fuse": args.cafa_use_fuse,
        "use_model": args.cafa_use_model,
        "bs": args.dp_bs,
        "lr": args.dp_lr
    }
    train_model_with_config(config, args.task_name, args.dp_fuse_base, args.mol_emd, args.protein_emd, args.dp_print,
                            args.max_no_improve, fuse_model=fuse_model, tune_mode=tune_mode)


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
