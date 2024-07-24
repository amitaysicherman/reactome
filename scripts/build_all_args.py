from common.data_types import PROT_UI_ORDER, MOL_UI_ORDER

CONFIGS = ["PROT", "NO", "COMP"]
PD_DATASET = ["DrugBank", "Davis", "KIBA"]
GO_TASKS = ["CC", "BP", "MF"]
MOL_TASKS = ["BACE", "ClinTox", "HIV", "SIDER"]
LOC_BIN = [0, 1]
DOWNSTREAM_TASKS = ["pd", "go", "loc", "mol"]


def downsteam_to_args(name):
    if name == "pd":
        return [({"db_dataset": d},d) for d in PD_DATASET]
    elif name == "go":
        all_go = {"dp_bs": 64, "dp_lr": 0.00005}
        return [({"go_task": g, **all_go},g) for g in GO_TASKS]
    elif name == "loc":
        return [({"loc_bin": l},l) for l in LOC_BIN]
    elif name == "mol":
        all_mol = {"dp_bs": 64, "dp_lr": 0.00005}
        return [({"mol_task": m, **all_mol},m) for m in MOL_TASKS]
    else:
        raise Exception("Unknown downstream task")


def config_to_args(name):
    if name == "PROT":
        return {"fuse_all_to_one": "protein", "fuse_n_layers": 1, "task_output_prefix": f"{name}_"}
    elif name == "NO":
        return {"fuse_all_to_one": "0", "fuse_n_layers": 1, "task_output_prefix": f"{name}_"}
    elif name == "COMP":
        return {"fuse_all_to_one": "protein", "fuse_n_layers": 4, "task_output_prefix": f"{name}_"}


for p in PROT_UI_ORDER:
    for m in MOL_UI_ORDER:
        for c in CONFIGS:
            for d in DOWNSTREAM_TASKS:
                config_args= config_to_args(c)
                for d_args,d_name in downsteam_to_args(d):
                    args_str=""
                    for k,v in {**config_args, **d_args}.items():
                        args_str+=f"--{k} {v} "

                    print(f" --name {c}-{d}-{d_name}-{p}-{m} --downstream_task cl --fuse_train_all 0 --dp_print 0 {args_str}")

