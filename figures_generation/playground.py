from common.utils import reaction_from_str
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from dataset.dataset_builder import get_reactions, get_reaction_entities
from dataset.index_manger import NodesIndexManager
from common.data_types import PROTEIN, MOLECULE
from common.path_manager import figures_path

types_in = [PROTEIN, MOLECULE]
index_manager = NodesIndexManager()
train_lines, valid_lines, test_lines = get_reactions(filter_dna=True)
all_i = []
for i, reactoion in tqdm(enumerate(train_lines)):
    entities = get_reaction_entities(reactoion, True)
    entities_names = [entity.get_db_identifier() for entity in entities]
    complexes = [entity.complex_id for entity in entities if entity.complex_id is not None]

    entities = [index_manager.name_to_node[name] for name in entities_names if
                index_manager.name_to_node[name].type in types_in]
    proteins_count = len(set([entity.index for entity in entities if entity.type == PROTEIN]))
    molecules_count = len(set([entity.index for entity in entities if entity.type == MOLECULE]))
    if 3 <= proteins_count <= 5 and 2 <= molecules_count <= 5:
        all_i.append(i)
    else:
        print("HERE")
# choose random 50 pairs
import random

pairs = [random.sample(all_i, 2) for _ in range(50)]
print(pairs)

print(len(all_i), all_i)
