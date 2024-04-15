from biopax_parser import Reaction, reaction_from_dict
from collections import defaultdict

locations = defaultdict(int)
entities = defaultdict(int)
catalyst_activities = defaultdict(int)
modifications = defaultdict(int)
input_file = "./data/items/reaction.txt"
with open(input_file) as f:
    lines = f.readlines()
reactions = []
for line in lines:
    reaction_dict = eval(line)
    reaction = reaction_from_dict(reaction_dict)
    for input in reaction.inputs:
        id = input.get_unique_id()
        entities[id] += 1
        locations[input.location] += 1
        for mod in input.modifications:
            modifications[mod] += 1
    for output in reaction.outputs:
        id = output.get_unique_id()
        entities[id] += 1
        locations[output.location] += 1
        for mod in output.modifications:
            modifications[mod] += 1
    for catalyst in reaction.catalysis:
        for entity in catalyst.entities:
            id = entity.get_unique_id()
            entities[id] += 1
            locations[entity.location] += 1
            for mod in entity.modifications:
                modifications[mod] += 1
        catalyst_activities[catalyst.activity] += 1

base_dir = "./data/items/"
with open(f"{base_dir}entities.txt", "w") as f:
    for k, v in entities.items():
        f.write(f"{k}:{v}\n")
with open(f"{base_dir}locations.txt", "w") as f:
    for k, v in locations.items():
        f.write(f"{k}:{v}\n")
with open(f"{base_dir}catalyst_activities.txt", "w") as f:
    for k, v in catalyst_activities.items():
        f.write(f"{k}:{v}\n")
with open(f"{base_dir}modifications.txt", "w") as f:
    for k, v in modifications.items():
        f.write(f"{k}:{v}\n")
