import glob
import os
from common.path_manager import model_path
from common.utils import get_best_gnn_cp
from tqdm import tqdm

for model_name in tqdm(os.listdir(model_path)):
    if not model_name.startswith("gnn_"):
        continue
    model_name = model_name.replace("gnn_", "")
    aug_data = model_name.split("_")[-1]

    best_model = get_best_gnn_cp(model_name, aug_data)
    print(best_model)
    for cp in tqdm(glob.glob(f"{model_path}/gnn_{model_name}/*")):
        print(cp)
        if cp != best_model:
            print("remove")
            # os.remove(cp)
