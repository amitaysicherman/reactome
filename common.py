# Description: This file contains common variables and functions used by other files in the project.
REACTION = "reaction"
COMPLEX = "complex"
UNKNOWN_ENTITY_TYPE = "?"
DNA = "dna"
PROTEIN = "protein"
MOLECULE = "molecule"
TEXT = "text"
EMBEDDING_DATA_TYPES = [DNA, PROTEIN, MOLECULE, TEXT]

LOCATION = "location"
DATA_TYPES = EMBEDDING_DATA_TYPES + [LOCATION] + [UNKNOWN_ENTITY_TYPE]


def db_to_type(db_name):
    db_name = db_name.lower()
    if db_name == "ensembl":
        return DNA
    elif db_name == "embl":
        return PROTEIN
    elif db_name == "uniprot" or db_name == "uniprot isoform":
        return PROTEIN
    elif db_name == "chebi":
        return MOLECULE
    elif db_name == "guide to pharmacology":
        return MOLECULE
    elif db_name == "go":
        return TEXT
    elif db_name == "text":
        return TEXT
    elif db_name == "ncbi nucleotide":
        return DNA
    elif db_name == "pubchem compound":
        return MOLECULE
    else:
        return UNKNOWN_ENTITY_TYPE
        #raise ValueError(f"Unknown database name: {db_name}")
