import dataclasses
from typing import List
from collections import defaultdict
from common.utils import reaction_from_str
from common.data_types import Entity, Reaction
import numpy as np
from dataclasses import dataclass


def get_change_entities(reaction: Reaction):
    input_unique_name_to_entity = {e.get_db_identifier() for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_db_identifier() for e in reaction.outputs}
    return input_unique_name_to_entity.symmetric_difference(output_unique_name_to_entity)


def is_move_reation(reaction: Reaction):
    input_unique_name_to_entity = {e.get_db_identifier(): e for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_db_identifier(): e for e in reaction.outputs}
    for input_key, input_entity in input_unique_name_to_entity.items():
        if input_key in output_unique_name_to_entity:
            if input_entity.location != output_unique_name_to_entity[input_key].location:
                return True
    return False


def is_modification_reation(reaction: Reaction):
    input_unique_name_to_entity = {e.get_db_identifier(): e for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_db_identifier(): e for e in reaction.outputs}
    for input_key, input_entity in input_unique_name_to_entity.items():
        if input_key in output_unique_name_to_entity:
            if input_entity.modifications != output_unique_name_to_entity[input_key].modifications:
                return True
    return False


def is_atp_adp_reation(reaction: Reaction):
    return "CHEBI:30616" in [e.db_id for e in reaction.inputs] and "CHEBI:456216" in [e.db_id for e in reaction.outputs]


def check_two_in_one(inputs_set_1, input_set_2, output_set):
    for input_1 in inputs_set_1:
        if input_1 not in output_set:
            return False
    for input_2 in input_set_2:
        if input_2 not in output_set:
            return False
    return True


def get_top_level_entities(entities: List[Entity]):
    complex_to_entities = defaultdict(list)
    for entity in entities:
        complex_to_entities[entity.complex_id].append(entity)
    top_level_entities = []
    for complex_id, complex_entities in complex_to_entities.items():
        if not complex_id:
            for entity in complex_entities:
                top_level_entities.append([entity.get_db_identifier()])
        else:
            top_level_entities.append([e.get_db_identifier() for e in complex_entities])
    return top_level_entities


def is_binding_reaction(reaction: Reaction):
    input_top_level_entities = get_top_level_entities(reaction.inputs)
    output_top_level_entities = get_top_level_entities(reaction.outputs)
    for input_entity_1 in input_top_level_entities:
        for input_entity_2 in input_top_level_entities:
            if input_entity_1 == input_entity_2:
                continue
            for output_entity in output_top_level_entities:
                if check_two_in_one(input_entity_1, input_entity_2, output_entity):
                    return True
    return False


def is_dissociation_reaction(reaction: Reaction):
    input_top_level_entities = get_top_level_entities(reaction.inputs)
    output_top_level_entities = get_top_level_entities(reaction.outputs)
    for output_entity_1 in output_top_level_entities:
        for output_entity_2 in output_top_level_entities:
            if output_entity_1 == output_entity_2:
                continue
            for input_entity in input_top_level_entities:
                if check_two_in_one(output_entity_1, output_entity_2, input_entity):
                    return True
    return False


def is_chemical_reaction(reaction: Reaction):
    for entity in reaction.inputs:
        if entity.db.lower() != "chebi":
            return False
    for entity in reaction.outputs:
        if entity.db.lower() != "chebi":
            return False
    input_entities = {e.get_db_identifier() for e in reaction.inputs}
    output_entities = {e.get_db_identifier() for e in reaction.outputs}
    return input_entities != output_entities


def is_dna_to_protein_reaction(reaction: Reaction):
    if len(reaction.inputs) != 1 or len(reaction.outputs) != 1:
        return False
    if reaction.inputs[0].db.lower() != "ensembl":
        return False
    if reaction.outputs[0].db.lower() != "uniprot":
        return False
    return True


def is_empty_output_reaction(reaction: Reaction):
    return len(reaction.outputs) == 0


def is_same_elements_reaction(reaction: Reaction):
    input_unique_name_to_entity = {e.get_db_identifier() for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_db_identifier() for e in reaction.outputs}
    return input_unique_name_to_entity == output_unique_name_to_entity


def is_unknown_database_reaction(reaction: Reaction):
    for entity in reaction.inputs:
        if entity.db == '0':
            return True
    for entity in reaction.outputs:
        if entity.db == '0':
            return True
    return False


@dataclass
class ReactionTag:
    move: bool = False
    modification: bool = False
    binding: bool = False
    dissociation: bool = False
    chemical: bool = False
    fake: bool = False

    def __str__(self):
        res = ""
        if self.move:
            res += "move "
        if self.modification:
            res += "modification "
        if self.binding:
            res += "binding "
        if self.dissociation:
            res += "dissociation "
        if self.chemical:
            res += "chemical "
        if self.fake:
            res += "fake "
        return res

    def get_num_tags(self):
        return len(dataclasses.astuple(self))

    def get_names(self, get_fake=False):
        if get_fake:
            return ["Move", "Modification", "Binding", "Dissociation", "Chemical", "Fake"]
        return ["Move", "Modification", "Binding", "Dissociation", "Chemical"]


def tag(reaction: Reaction) -> ReactionTag:
    return ReactionTag(
        move=is_move_reation(reaction),
        modification=is_modification_reation(reaction),
        binding=is_binding_reaction(reaction),
        dissociation=is_dissociation_reaction(reaction),
        chemical=is_chemical_reaction(reaction),
    )


if __name__ == '__main__':
    root = "data/items"
    with open(f'{root}/reaction.txt') as f:
        lines = f.readlines()
    tags = []
    for line in lines:
        reaction = reaction_from_str(line)
        new_tag = str(tag(reaction)).split()
        if len(new_tag) == 0:
            tags.append('')
        else:
            tags.extend(new_tag)

    print(tags.count("") / len(tags), tags.count("move") / len(tags), tags.count("modification") / len(tags),
          tags.count("binding") / len(tags), tags.count("dissociation") / len(tags), tags.count("chemical") / len(tags))
    print(np.unique(tags, return_counts=True))
