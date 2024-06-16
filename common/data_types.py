import dataclasses
import datetime
from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    name: str
    db: str
    db_id: str
    location: str
    modifications: tuple = ()
    complex_id: int = 0

    def get_db_identifier(self):
        return self.db + "@" + self.db_id

    def to_dict(self):
        return {
            "name": self.name,
            "db": self.db,
            "db_id": self.db_id,
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


class Reaction:
    def __init__(self, name, inputs: List[Entity], outputs: List[Entity], catalysis: List[CatalystOBJ],
                 date: datetime.date, reactome_id: str, biological_process:List[str]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.catalysis = catalysis
        self.date = date
        self.reactome_id = reactome_id
        self.biological_process = biological_process

    def to_dict(self):
        return {
            "name": self.name,
            "inputs": [e.to_dict() for e in self.inputs],
            "outputs": [e.to_dict() for e in self.outputs],
            "catalysis": [c.to_dict() for c in self.catalysis],
            "date": f'{self.date.year}_{self.date.month}_{self.date.day}',
            "reactome_id": self.reactome_id,
            "biological_process": "_".join(self.biological_process)
        }


REACTION = "reaction"
COMPLEX = "complex"
UNKNOWN_ENTITY_TYPE = "_"
DNA = "dna"
PROTEIN = "protein"
MOLECULE = "molecule"
TEXT = "text"
EMBEDDING_DATA_TYPES = [DNA, PROTEIN, MOLECULE, TEXT]
LOCATION = "location"
DATA_TYPES = EMBEDDING_DATA_TYPES + [LOCATION] + [UNKNOWN_ENTITY_TYPE]
BIOLOGICAL_PROCESS = "bp"

@dataclasses.dataclass
class NodeTypes:
    reaction = REACTION
    complex = COMPLEX
    location = LOCATION
    protein = PROTEIN
    dna = DNA
    molecule = MOLECULE
    text = TEXT


@dataclasses.dataclass
class EdgeTypes:
    # location
    location_self_loop = (NodeTypes.location, "location_self_loop", NodeTypes.location)
    location_to_protein = (NodeTypes.location, "location_to_protein", NodeTypes.protein)
    location_to_dna = (NodeTypes.location, "location_to_dna", NodeTypes.dna)
    location_to_molecule = (NodeTypes.location, "location_to_molecule", NodeTypes.molecule)

    # modification
    modification_self_loop = (NodeTypes.text, "modification_self_loop", NodeTypes.text)
    modification_to_protein = (NodeTypes.text, "modification_to_protein", NodeTypes.protein)

    # catalysis_activity
    catalysis_activity_self_loop = (NodeTypes.text, "catalysis_activity_self_loop", NodeTypes.text)
    catalysis_activity_to_reaction = (NodeTypes.text, "catalysis_activity_to_reaction", NodeTypes.reaction)

    # catalysis
    catalysis_protein_to_reaction = (NodeTypes.protein, "catalysis_protein_to_reaction", NodeTypes.reaction)
    catalysis_dna_to_reaction = (NodeTypes.dna, "catalysis_dna_to_reaction", NodeTypes.reaction)
    catalysis_molecule_to_reaction = (NodeTypes.molecule, "catalysis_molecule_to_reaction", NodeTypes.reaction)

    # reaction input
    protein_to_reaction = (NodeTypes.protein, "protein_to_reaction", NodeTypes.reaction)
    dna_to_reaction = (NodeTypes.dna, "dna_to_reaction", NodeTypes.reaction)
    molecule_to_reaction = (NodeTypes.molecule, "molecule_to_reaction", NodeTypes.reaction)

    # complex
    complex_to_reaction = (NodeTypes.complex, "complex_to_reaction", NodeTypes.reaction)
    protein_to_complex = (NodeTypes.protein, "protein_to_complex", NodeTypes.complex)
    dna_to_complex = (NodeTypes.dna, "dna_to_complex", NodeTypes.complex)
    molecule_to_complex = (NodeTypes.molecule, "molecule_to_complex", NodeTypes.complex)
    catalysis_protein_to_complex = (NodeTypes.protein, "catalysis_protein_to_complex", NodeTypes.complex)
    catalysis_dna_to_complex = (NodeTypes.dna, "catalysis_dna_to_complex", NodeTypes.complex)
    catalysis_molecule_to_complex = (NodeTypes.molecule, "catalysis_molecule_to_complex", NodeTypes.complex)

    def get(self, str):
        return getattr(self, str)

    def get_by_src_dst(self, src, dst, is_catalysis=False):
        if src == NodeTypes.text:
            if dst == NodeTypes.text:
                if is_catalysis:
                    text = f'catalysis_activity_self_loop'
                else:
                    text = f'modification_self_loop'
            elif is_catalysis:
                text = f'catalysis_activity_to_{dst}'
            else:
                text = f'modification_to_{dst}'
        elif src == dst and src == NodeTypes.location:
            text = f'location_self_loop'
        elif is_catalysis:
            text = f'catalysis_{src}_to_{dst}'
        else:
            text = f'{src}_to_{dst}'
        return self.get(text)


REAL = "real"
FAKE_LOCATION_ALL = "fake_location"
FAKE_PROTEIN = "fake_protein"
FAKE_MOLECULE = "fake_molecule"