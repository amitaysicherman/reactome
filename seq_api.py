import requests
from libchebipy._chebi_entity import ChebiEntity
from tqdm import tqdm


def get_req(url: str, to_json=False):
    response = requests.get(url)
    if response.status_code == 200:
        if to_json:
            return response.json()
        return response.text
    else:
        print(f"Failed to retrieve url: {url}")
        return ""


def from_second_line(seq):
    seq = seq.split("\n")
    if len(seq) < 2:
        return ""
    return "".join(seq[1:])


def db_to_type(db_name):
    if db_name == "ensembl":
        return "DNA"
    elif db_name == "embl":
        return "Protein"
    elif db_name == "uniprot":
        return "Protein"
    elif db_name == "chebi":
        return "Molecule"
    elif db_name == "guide to pharmacology":
        return "Molecule"
    elif db_name == "go":
        return "Text"
    else:
        return "?"


def get_sequence(id, db_name):
    db_name = db_name.lower()
    if db_name == "ensembl":
        seq = get_req(f"https://rest.ensembl.org/sequence/id/{id}?content-type=text/plain;species=human")
    elif db_name == "embl":
        seq = get_req(f"https://www.ebi.ac.uk/ena/browser/api/fasta/{id}")
        seq = from_second_line(seq)
    elif db_name == "uniprot":
        seq = get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")
        seq = from_second_line(seq)
    elif db_name == "chebi":
        chebi_entity = ChebiEntity(id)
        seq = chebi_entity.get_smiles()
    elif db_name == "guide to pharmacology":
        json_req = get_req(f"https://www.guidetopharmacology.org/services/ligands/{id}/structure", to_json=True)
        if "smiles" not in json_req:
            seq = ""
        else:
            seq = json_req["smiles"]
    elif db_name == "go":
        json_req = get_req(f"https://api.geneontology.org/api/ontology/term/{id}", to_json=True)
        print(json_req)
        seq = ""
        if "label" in json_req:
            seq += json_req["label"] + ". "
        if "definition" in json_req:
            seq += json_req["definition"]
        return seq
    else:
        # print("Unknown database")
        seq = ""
    return seq


with open("./data/items/entities.txt") as f:
    entities = f.read().split("\n")

seqs = []
db_types = []
for entity in tqdm(entities):
    if len(entity.split("_")) != 2:
        print("entity:", entity)
        db_types.append("?")
        seqs.append("")
        continue
    db_name, id_ = entity.split("_")
    db_name = db_name.lower()
    db_name = db_name.replace(" isoform", "")
    *id_, _ = id_.split(":")
    id_ = ":".join(id_)
    seqs.append(get_sequence(id_, db_name))
    db_types.append(db_to_type(db_name))

with open("./data/items/entities_sequences.txt", "w") as f:
    for t, s in zip(db_types, seqs):
        f.write(f"{t} {s}\n")

# with open("./data/items/catalyst_activities.txt") as f:
#     catalyst_activities = f.read().split("\n")
#
#
# seqs = []
# for go_activity in tqdm(catalyst_activities):
#     *go_activity, _ = go_activity.split(":")
#     go_activity = ":".join(go_activity)
#     seqs.append(get_sequence(go_activity, "go"))
#
# with open("./data/items/catalyst_activities_sequences.txt", "w") as f:
#     for s in seqs:
#         f.write(f"Text {s}\n")
