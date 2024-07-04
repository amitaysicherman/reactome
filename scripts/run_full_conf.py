from common.args_manager import get_args
import os


def args_to_string(args):
    res = ""
    for arg in vars(args):
        if getattr(args, arg) == "":
            continue
        res += f"--{arg} {getattr(args, arg)} "
    return res


args = get_args()

# run fusing
cmd = f"python3 model/contrastive_learning.py {args_to_string(args)}"
print("Running command: ", cmd)
os.system(cmd)

# run drug protein
args.dp_print = 1
args.dp_fuse_base = f"{args.dp_fuse_base}_best"
configs = [{"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 1, "dp_p_model": 1},
           {"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 0, "dp_p_model": 0},
           {"dp_m_fuse": 0, "dp_p_fuse": 0, "dp_m_model": 1, "dp_p_model": 1},
           {"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 0, "dp_p_model": 1},
           {"dp_m_fuse": 1, "dp_p_fuse": 1, "dp_m_model": 1, "dp_p_model": 0},
           {"dp_m_fuse": 0, "dp_p_fuse": 1, "dp_m_model": 1, "dp_p_model": 1},
           {"dp_m_fuse": 1, "dp_p_fuse": 0, "dp_m_model": 1, "dp_p_model": 1}]
for config in configs:
    for random_seed in range(42, 52):
        args.random_seed = random_seed
        for key, value in config.items():
            setattr(args, key, value)
        cmd = f"python protein_drug/train_eval.py {args_to_string(args)}"
        print("Running command: ", cmd)
        os.system(cmd)
