from common.args_manager import get_args
import os


def args_to_string(args):
    res = ""
    for arg in vars(args):
        val = getattr(args, arg)
        if val == "":
            continue
        if type(val) == list:
            res += f"--{arg} {' '.join([str(v) for v in val])} "
        else:
            res += f"--{arg} {val} "

    return res


args = get_args()

# run fusing
cmd = f"python3 model/contrastive_learning.py {args_to_string(args)}"
print("Running command: ", cmd)
os.system(cmd)

args.dp_print = 1

if args.downstream_task == "pd":
    # run drug protein
    configs = [{"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 1, "dp_p_model": 1},
               {"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 0, "dp_p_model": 0},
               {"dp_m_fuse": 0, "dp_p_fuse": 0, "dp_m_model": 1, "dp_p_model": 1},
               {"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 0, "dp_p_model": 1},
               {"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 1, "dp_p_model": 0},
               {"dp_m_fuse": 0, "dp_p_fuse": 1, "dp_m_model": 1, "dp_p_model": 1},
               {"dp_m_fuse": 1, "dp_p_fuse": 0, "dp_m_model": 1, "dp_p_model": 1}]

    base_cmd = "python protein_drug/train_eval.py"
elif args.downstream_task == "rrf":
    configs = [{"dp_print": 1}]  # no effect. just to keep the same format
    base_cmd = f"python reaction_real_fake/train_eval.py"
else:
    # run localization
    configs = [{"cafa_use_fuse": 1, "cafa_use_model": 1},
               {"cafa_use_fuse": 1, "cafa_use_model": 0},
               {"cafa_use_fuse": 0, "cafa_use_model": 1}]
    if args.downstream_task == "go":
        base_cmd = "python GO/train_eval.py"
    elif args.downstream_task == "loc":
        base_cmd = "python localization/train_eval.py"
    else:
        raise Exception("Unknown downstream task")

for random_seed in range(42, 52):
    args.random_seed = random_seed
    for config in configs:
        for key, value in config.items():
            setattr(args, key, value)
        cmd = f"{base_cmd} {args_to_string(args)}"
        print("Running command: ", cmd)
        os.system(cmd)
