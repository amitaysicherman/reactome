from enum import Enum
from typing import Dict


class CellularComponent(Enum):
    CYTOSOL = 1
    NUCLEOPLASM = 2
    MITOCHONDRIAL_MATRIX = 3
    MITOCHONDRIAL_INNER_MEMBRANE = 4


class PhysicalEntity:
    def __init__(self, name, cellular_component: CellularComponent):
        self.name = name
        self.cellular_component = cellular_component
        self.modification = None
        self.binding_entities = []

    def move_to(self, cellular_component):
        self.cellular_component = cellular_component

    def modify(self, modification: str):
        self.modification = modification

    def bind(self, entity):
        self.binding_entities.append(entity)


class BiologicalCellularEnvironment:
    def __init__(self):
        self.physical_entities: Dict[str, PhysicalEntity] = dict()
        self.catalysts = []

    def add_physical_entity(self, entity: PhysicalEntity):
        self.physical_entities[entity.name] = entity

    def get_physical_entity(self, name: str):
        return self.physical_entities[name]

    def remove_physical_entity(self, name: str):
        del self.physical_entities[name]

    def add_catalyst(self, catalyst):
        self.catalysts.append(catalyst)


class Catalyst:
    def __init__(self, entity: PhysicalEntity, activity: str):
        self.entity = entity
        self.activity = activity


########################################
# Reaction 1
########################################
cell = BiologicalCellularEnvironment()
input_1 = PhysicalEntity("ATP", CellularComponent.CYTOSOL)
cell.add_physical_entity(input_1)
input_2 = PhysicalEntity("CLF1", CellularComponent.CYTOSOL)
cell.add_physical_entity(input_2)
catalyst_entity = PhysicalEntity("LIMK1", CellularComponent.CYTOSOL)
catalyst_entity.modify(
    "A protein modification that effectively converts an L-threonine residue to O-phospho-L-threonine")
catalyst = Catalyst(catalyst_entity, "Protein serine/threonine kinase activity")
cell.add_catalyst(catalyst)

# Apply reaction On the BiologicalCellularEnvironment:
cell.remove_physical_entity(input_1.name)
output_1 = PhysicalEntity("ADT", CellularComponent.CYTOSOL)
cell.add_physical_entity(output_1)
cell.get_physical_entity(input_2.name).modify(
    "A protein modification that effectively converts an L-serine residue to O-phospho-L-serine")

########################################
# Reaction 2
########################################
cell = BiologicalCellularEnvironment()
input_1 = PhysicalEntity("LHX2", CellularComponent.NUCLEOPLASM)
cell.add_physical_entity(input_1)
input_2 = PhysicalEntity("ROBO1", CellularComponent.NUCLEOPLASM)
cell.add_physical_entity(input_2)

# Apply reaction On the BiologicalCellularEnvironment:
cell.get_physical_entity(input_1.name).bind(input_2)
cell.get_physical_entity(input_2.name).bind(input_1)

########################################
# Reaction 3
########################################
cell = BiologicalCellularEnvironment()
input_1 = PhysicalEntity("2-OA", CellularComponent.MITOCHONDRIAL_MATRIX)
cell.add_physical_entity(input_1)
input_2 = PhysicalEntity("2OG", CellularComponent.CYTOSOL)
cell.add_physical_entity(input_2)
catalyst_entity = PhysicalEntity("SLC25A21", CellularComponent.MITOCHONDRIAL_INNER_MEMBRANE)
catalyst = Catalyst(catalyst_entity, "Alpha-ketoglutarate transmembrane transporter activity")

# Apply reaction On the BiologicalCellularEnvironment:
cell.get_physical_entity(input_1.name).move_to(CellularComponent.CYTOSOL)
cell.get_physical_entity(input_2.name).move_to(CellularComponent.MITOCHONDRIAL_MATRIX)
