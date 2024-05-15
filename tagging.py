import dataclasses
from typing import List
from collections import defaultdict
from biopax_parser import reaction_from_str, Reaction, Entity
import pandas as pd
import numpy as np
from dataclasses import dataclass


def get_change_entities(reaction: Reaction):
    input_unique_name_to_entity = {e.get_unique_id() for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_unique_id() for e in reaction.outputs}
    return input_unique_name_to_entity.symmetric_difference(output_unique_name_to_entity)


def is_move_reation(reaction: Reaction):
    input_unique_name_to_entity = {e.get_unique_id(): e for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_unique_id(): e for e in reaction.outputs}
    for input_key, input_entity in input_unique_name_to_entity.items():
        if input_key in output_unique_name_to_entity:
            if input_entity.location != output_unique_name_to_entity[input_key].location:
                return True
    return False


def is_modification_reation(reaction: Reaction):
    input_unique_name_to_entity = {e.get_unique_id(): e for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_unique_id(): e for e in reaction.outputs}
    for input_key, input_entity in input_unique_name_to_entity.items():
        if input_key in output_unique_name_to_entity:
            if input_entity.modifications != output_unique_name_to_entity[input_key].modifications:
                return True
    return False


def is_atp_adp_reation(reaction: Reaction):
    return "CHEBI:30616" in [e.id for e in reaction.inputs] and "CHEBI:456216" in [e.id for e in reaction.outputs]


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
                top_level_entities.append([entity.get_unique_id()])
        else:
            top_level_entities.append([e.get_unique_id() for e in complex_entities])
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
    input_entities = {e.get_unique_id() for e in reaction.inputs}
    output_entities = {e.get_unique_id() for e in reaction.outputs}
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
    input_unique_name_to_entity = {e.get_unique_id() for e in reaction.inputs}
    output_unique_name_to_entity = {e.get_unique_id() for e in reaction.outputs}
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
    # atp_adp: bool = False
    binding: bool = False
    dissociation: bool = False
    chemical: bool = False
    # dna_to_protein: bool = False
    # empty_output: bool = False
    # same_elements: bool = False
    fake: bool = False

    def __str__(self):
        res = ""
        if self.move:
            res += "move "
        if self.modification:
            res += "modification "
        # if self.atp_adp:
        #     res += "atp_adp "
        if self.binding:
            res += "binding "
        if self.dissociation:
            res += "dissociation "
        if self.chemical:
            res += "chemical "
        # if self.dna_to_protein:
        #     res += "dna_to_protein "
        # if self.empty_output:
        #     res += "empty_output "
        # if self.same_elements:
        #     res += "same_elements "
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
        # atp_adp=is_atp_adp_reation(reaction),
        binding=is_binding_reaction(reaction),
        dissociation=is_dissociation_reaction(reaction),
        chemical=is_chemical_reaction(reaction),
        # dna_to_protein=is_dna_to_protein_reaction(reaction),
        # empty_output=is_empty_output_reaction(reaction),
        # same_elements=is_same_elements_reaction(reaction),
    )


if __name__ == '__main__':
    root = "data/items"
    with open(f'{root}/reaction.txt') as f:
        lines = f.readlines()
    # names = []
    # is_move = []
    # is_modification = []
    # is_atp_adp = []
    # is_binding = []
    # is_dissociation = []
    # is_chemical = []
    # is_dna_to_protein = []
    # is_empty_output = []
    # is_same_elements = []
    # is_unknown_database = []
    tags = []
    for line in lines:
        reaction = reaction_from_str(line)
        # print(reaction.name, len(reaction.inputs), sum([len(c.entities) for c in reaction.catalysis]),
        #      str(tag(reaction)).split())

        new_tag = str(tag(reaction)).split()
        if len(new_tag) == 0:
            tags.append('')
        else:
            tags.extend(new_tag)  # names.append(reaction.name)
        # is_move.append(is_move_reation(reaction))
        # is_modification.append(is_modification_reation(reaction))
        # is_atp_adp.append(is_atp_adp_reation(reaction))
        # is_binding.append(is_binding_reaction(reaction))
        # is_dissociation.append(is_dissociation_reaction(reaction))
        # is_chemical.append(is_chemical_reaction(reaction))
        # is_dna_to_protein.append(is_dna_to_protein_reaction(reaction))
        # is_empty_output.append(is_empty_output_reaction(reaction))
        # is_same_elements.append(is_same_elements_reaction(reaction))
        # is_unknown_database.append(is_unknown_database_reaction(reaction))
        #
        # print("move", np.mean(is_move), np.sum(is_move))
        # print("modification", np.mean(is_modification), np.sum(is_modification))
        # print("atp_adp", np.mean(is_atp_adp), np.sum(is_atp_adp))
        # print("binding", np.mean(is_binding), np.sum(is_binding))
        # print("dissociation", np.mean(is_dissociation), np.sum(is_dissociation))
        # print("chemical", np.mean(is_chemical), np.sum(is_chemical))
        # print("dna_to_protein", np.mean(is_dna_to_protein), np.sum(is_dna_to_protein))
        # print("empty_output", np.mean(is_empty_output), np.sum(is_empty_output))
        # print("same_elements", np.mean(is_same_elements), np.sum(is_same_elements))
        # print("unknown_database", np.mean(is_unknown_database), np.sum(is_unknown_database))
        #
        # results = pd.DataFrame(
        #     {'is_move': is_move, 'is_modification': is_modification, 'is_binding': is_binding,
        #      'is_dissociation': is_dissociation, 'is_chemical': is_chemical, 'is_dna_to_protein': is_dna_to_protein,
        #      'is_empty_output': is_empty_output, 'is_same_elements': is_same_elements,
        #      'is_unknown_database': is_unknown_database})
        # results.index = names
        # print(results)
        # empty_res = results[results.T.sum() == 0]
        # print(empty_res.index)
        # counts = results.T.sum().value_counts()
        # print(counts)
        # print(results[results.index == 'miR-211 RISC binds POU3F2 mRNA'].T)
        # print(results[results.index == 'CDT1-mediated formation of MCM2-7 double hexamer at the replication origin'].T)
    print(tags.count("") / len(tags), tags.count("move") / len(tags), tags.count("modification") / len(tags),
          tags.count("binding") / len(tags), tags.count("dissociation") / len(tags), tags.count("chemical") / len(tags))
    print(np.unique(tags, return_counts=True))
