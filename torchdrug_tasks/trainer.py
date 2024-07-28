import os
import torch
from common.data_types import Config
from common.path_manager import scores_path
from torchdrug_tasks.dataset import get_dataloaders
from torchdrug_tasks.tasks import name_to_task, Task

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_epoch(model, loader, optimizer, criterion, metric, part):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    for mols, labels in loader:
        mols = mols.to(device).float()
        optimizer.zero_grad()
        labels = labels.float().to(device)
        output = model(mols)
        loss = criterion(output, labels)
        if part == "train":
            loss.backward()
            optimizer.step()
        output = torch.sigmoid(output).detach()
        reals.append(labels)
        preds.append(output)
    if part != "train":
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        score = metric(reals.flatten(), preds.flatten())
        return score
    else:
        return 0


def get_model_from_task(task: Task, dataset, conf, **kwargs):
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
    return model_class(input_dim_1, dtype_1, input_dim_2, dtype_2, output_dim, conf, **kwargs)


def main(args, fuse_model=None):
    fuse_base = args.dp_fuse_base
    use_fuse = bool(args.cafa_use_fuse)
    use_model = bool(args.cafa_use_model)
    bs = args.dp_bs
    lr = args.dp_lr
    mol_emd = args.mol_emd
    protein_emd = args.protein_emd
    task_name = args.task_name
    train_loader, valid_loader, test_loader = get_dataloaders(task_name, mol_emd, protein_emd, bs)

    task = name_to_task[task_name]
    metric = task.metric
    criterion = task.criterion()
    if use_fuse and use_model:
        conf = Config.both
    elif use_fuse:
        conf = Config.our
    else:
        conf = Config.PRE
    model = get_model_from_task(task, train_loader.dataset, conf, fuse_base=fuse_base, fuse_model=fuse_model)

    if args.dp_print:
        print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    no_improve = 0
    best_valid_score = 0
    best_test_score = 0
    for epoch in range(250):
        train_score = run_epoch(model, train_loader, optimizer, criterion, metric, "train")
        with torch.no_grad():
            val_score = run_epoch(model, valid_loader, optimizer, criterion, metric, "val")
            test_score = run_epoch(model, test_loader, optimizer, criterion, metric, "test")

        if args.dp_print:
            print(epoch, train_score, val_score, test_score)
        if val_score > best_valid_score:
            best_valid_score = val_score
            best_test_score = test_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.max_no_improve:
                break

    if args.dp_print:
        print("Best Test scores\n", best_test_score)
        task_output_prefix = args.task_output_prefix
        output_file = f"{scores_path}/{task_output_prefix}torchdrug.csv"
        if not os.path.exists(output_file):
            names = "name,mol,prot,conf,prefix,score\n"
            with open(output_file, "w") as f:
                f.write(names)
        with open(output_file, "a") as f:
            f.write(f'{args.name},{mol_emd},{protein_emd},{conf},{task_output_prefix},{best_test_score}\n')
    return best_valid_score, best_test_score


if __name__ == '__main__':
    from common.args_manager import get_args

    main(get_args())
