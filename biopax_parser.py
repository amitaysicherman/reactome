import os.path
from dataclasses import dataclass
from typing import List
import pybiopax
from pybiopax.biopax import BiochemicalReaction
from tqdm import tqdm
import datetime

max_complex_id = 1


@dataclass
class Entity:
    name: str
    db: str
    id: str
    location: str
    modifications: tuple = ()
    complex_id: int = 0

    def get_unique_id(self):
        return self.db + "@" + self.id

    def to_dict(self):
        return {
            "name": self.name,
            "db": self.db,
            "id": self.id,
            "location": self.location,
            "modifications": self.modifications,
            "complex_id": self.complex_id
        }


@dataclass
class CatalystOBJ:
    entities: List[Entity]
    activity: str

    def to_dict(self):
        return {
            "entities": [e.to_dict() for e in self.entities],
            "activity": self.activity
        }


def catalyst_from_dict(d: dict) -> CatalystOBJ:
    entities = [Entity(**e) for e in d["entities"]]
    activity = d["activity"]
    return CatalystOBJ(entities, activity)


class Reaction:
    def __init__(self, name, inputs: List[Entity], outputs: List[Entity], catalysis: List[CatalystOBJ],
                 date: datetime.date, reactome_id: str):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.catalysis = catalysis
        self.date = date
        self.reactome_id = reactome_id

    def to_dict(self):
        return {
            "name": self.name,
            "inputs": [e.to_dict() for e in self.inputs],
            "outputs": [e.to_dict() for e in self.outputs],
            "catalysis": [c.to_dict() for c in self.catalysis],
            "date": f'{self.date.year}_{self.date.month}_{self.date.day}',
            "reactome_id": self.reactome_id
        }

    def to_tuple(self):
        seq = []
        entities = self.inputs + self.outputs + [e for c in self.catalysis for e in c.entities]
        for e in entities:
            seq.append(e.get_unique_id())
            seq.append(e.location)
            seq.append(e.modifications)
        for c in self.catalysis:
            seq.append(c.activity)
        seq = sorted([str(s) for s in seq if s])
        return tuple(seq)

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())


def reaction_from_dict(d: dict) -> Reaction:
    name = d["name"]
    inputs = [Entity(**e) for e in d["inputs"]]
    outputs = [Entity(**e) for e in d["outputs"]]
    catalysis = [catalyst_from_dict(c) for c in d["catalysis"]]
    year, month, day = d["date"].split("_")
    date = datetime.date(int(year), int(month), int(day))
    reactome_id = d["reactome_id"]
    return Reaction(name, inputs, outputs, catalysis, date, reactome_id)


def reaction_from_str(s: str) -> Reaction:
    return reaction_from_dict(eval(s))


def feature_parser(feature) -> str:
    if isinstance(feature, pybiopax.biopax.FragmentFeature):
        return ""
    if isinstance(feature, pybiopax.biopax.ModificationFeature):
        if not hasattr(feature, "modification_type") or feature.modification_type is None:
            return feature.comment[0].split(" ")[0]
        return feature.modification_type.term[0]


def element_parser(element: pybiopax.biopax.PhysicalEntity, complex_location=None, complex_id=0):
    if not hasattr(element, "entity_reference") or not hasattr(element.entity_reference, "xref"):
        if hasattr(element, "xref"):
            for xref in element.xref:
                if "Reactome" not in xref.db:
                    ref_db = xref.db
                    ref_id = xref.id
                    break
            else:
                ref_db = "0"
                ref_id = element.display_name

    elif len(element.entity_reference.xref) > 1:
        print(len(element.entity_reference.xref), "xrefs")
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id
    else:
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id

    name = element.display_name
    if complex_location:
        location = complex_location
    else:
        if not hasattr(element, "cellular_location") or not hasattr(element.cellular_location, "term"):
            location = ""
        else:
            location = element.cellular_location.term[0]

    features = list(element.feature)
    modifications = [feature_parser(f) for f in features]
    modifications = [f for f in modifications if f != ""]
    modifications = tuple(modifications)
    return Entity(name, ref_db, ref_id, location, modifications, complex_id)


def add_protein_or_complex(entity, complex_location=None, complex_id=0):
    elements = []
    if entity.member_physical_entity:
        entity = entity.member_physical_entity[0]  # TODO: get all set elements

    if isinstance(entity, pybiopax.biopax.Complex):

        if complex_location is None:
            global max_complex_id
            complex_id = max_complex_id
            max_complex_id += 1
            complex_location = entity.cellular_location.term[0]
        for entity in entity.component:
            elements.extend(add_protein_or_complex(entity, complex_location, complex_id))
    elif isinstance(entity, pybiopax.biopax.PhysicalEntity):
        elements.append(element_parser(entity, complex_location, complex_id))

    else:
        print("Unknown entity", type(entity))
    return elements


def catalysis_parser(catalysis_list: List[pybiopax.biopax.Catalysis]) -> List[CatalystOBJ]:
    results = []
    for catalysis in catalysis_list:
        assert len(catalysis.controller) == 1, "More than one controller"
        catalysis_entities = add_protein_or_complex(catalysis.controller[0])
        catalysis_activity = catalysis.xref[0].id
        results.append(CatalystOBJ(catalysis_entities, catalysis_activity))
    return results


def get_reactome_id(reaction: BiochemicalReaction) -> str:
    if not hasattr(reaction, "xref"):
        return "0"
    for xref in reaction.xref:
        if "Reactome" in xref.db:
            return xref.id
    return "0"


def get_reaction_date(reaction: BiochemicalReaction, format='%Y-%m-%d',
                      default_date=datetime.date(1970, 1, 1)) -> datetime.date:
    if not hasattr(reaction, "comment"):
        return default_date
    comments = [c.split(", ")[-1].split(" ")[0] for c in reaction.comment if "Authored" in c]
    if not comments:
        return default_date
    date_str = comments[0]
    try:
        date = datetime.datetime.strptime(date_str, format).date()
    except:
        date = default_date
        print(f"Error in date: {date_str}")

    return date


if __name__ == "__main__":

    write_output = True

    input_file = '/home/amitay/PycharmProjects/reactome/data/biopax/Homo_sapiens.owl'  # Homo_sapiens # R-HSA-3928608_level3.owl'  # R-HSA-8850529_level3.owl'  # '  # Homo_sapiens.owl'
    if write_output:
        output_file = "data/items/reaction.txt"
        if os.path.exists(output_file):
            os.remove(output_file)
    model = pybiopax.model_from_owl_file(input_file)
    reactions = list(model.get_objects_by_type(pybiopax.biopax.BiochemicalReaction))
    all_catalysis = list(model.get_objects_by_type(pybiopax.biopax.Catalysis))
    print(len(reactions))
    for i, reaction in tqdm(enumerate(reactions)):
        assert reaction.conversion_direction == "LEFT-TO-RIGHT"
        left_elements = []
        for entity in reaction.left:
            left_elements.extend(add_protein_or_complex(entity))

        right_elements = []
        for entity in reaction.right:
            right_elements.extend(add_protein_or_complex(entity))
        catalys_activities = [c for c in all_catalysis if c.controlled == reaction]
        catalys_activities = catalysis_parser(catalys_activities)
        date = get_reaction_date(reaction)
        reactome_id = get_reactome_id(reaction)
        reaction_obj = Reaction(reaction.name[0], left_elements, right_elements, catalys_activities, date, reactome_id)
        if write_output:
            with open(output_file, "a") as f:
                f.write(f'{reaction_obj.to_dict()}\n')
