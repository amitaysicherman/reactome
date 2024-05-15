from biopax_parser import reaction_from_str
from common import DATA_TYPES, db_to_type, LOCATION, TEXT
from collections import defaultdict
from tqdm import tqdm

indexes = {dt: defaultdict(int) for dt in DATA_TYPES}


base_dir = "./data/mus/"

input_file = f"{base_dir}/reaction.txt"

with open(input_file) as f:
    lines = f.readlines()
reactions = []
for line in tqdm(lines):
    reaction = reaction_from_str(line)
    catalyst_entities = sum([c.entities for c in reaction.catalysis], [])
    all_entities = reaction.inputs + reaction.outputs + catalyst_entities
    for entity in all_entities:
        dtype = db_to_type(entity.db)
        indexes[dtype][entity.get_unique_id()] += 1
        indexes[LOCATION][entity.location] += 1
        for mod in entity.modifications:
            mod = f"TEXT@{mod}"
            indexes[TEXT][mod] += 1
    for catalyst in reaction.catalysis:
        catalyst_activity = f"GO@{catalyst.activity}"
        indexes[TEXT][catalyst_activity] += 1

for k, v in indexes.items():
    with open(f"{base_dir}{k}.txt", "w") as f:
        for k, v in v.items():
            f.write(f"{k}@{v}\n")
