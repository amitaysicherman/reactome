import pathlib
import os

BASE_DIR = str(pathlib.Path(__file__).parent.parent.resolve())

data_path = os.path.join(BASE_DIR, "data")
item_path = os.path.join(data_path, "items")
figures_path = os.path.join(data_path, "figures")
scores_path = os.path.join(data_path, "scores")
model_path = os.path.join(data_path, "models_checkpoints")
reactions_file = os.path.join(item_path, "reaction.txt")
fuse_path = os.path.join(model_path, "fuse")
