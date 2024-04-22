from common import db_to_type, EMBEDDING_DATA_TYPES
import requests
from libchebipy._chebi_entity import ChebiEntity
from tqdm import tqdm


def get_req(url: str, to_json=False):
    for i in range(3):
        response = requests.get(url)
        if response.status_code == 200:
            if to_json:
                return response.json()
            return response.text
        else:
            print(f"Failed to retrieve url ({i}): {url} ")
    if to_json:
        return {}
    return ""


def from_second_line(seq):
    seq = seq.split("\n")
    if len(seq) < 2:
        return ""
    return "".join(seq[1:])


def get_sequence(identifier, db_name):
    db_name = db_name.lower()
    default_seq = ""

    # Define URLs and processing for each database
    db_handlers = {
        "ensembl": lambda id: get_req(
            f"https://rest.ensembl.org/sequence/id/{id}?content-type=text/plain;species=human"),
        "embl": lambda id: from_second_line(get_req(f"https://www.ebi.ac.uk/ena/browser/api/fasta/{id}")),
        "uniprot": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "uniprot isoform": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "chebi": lambda id: ChebiEntity(id).get_smiles() or default_seq,
        "guide to pharmacology": lambda id: get_req(
            f"https://www.guidetopharmacology.org/services/ligands/{id}/structure",
            to_json=True
        ).get("smiles", default_seq),
        "go": lambda id: parse_go_response(
            get_req(f"https://api.geneontology.org/api/ontology/term/{id}", to_json=True)
        ),
        "ncbi nucleotide": lambda id: from_second_line(
            get_req(f"https://www.ncbi.nlm.nih.gov/nuccore/{id}?report=fasta&log$=seqview&format=text")),
        "pubchem compound": lambda id: get_req(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/CanonicalSMILES/JSON",
            to_json=True
        ).get("CanonicalSMILES", default_seq),
        "text": lambda id: id
    }

    def parse_go_response(json_resp):
        label = json_resp.get("label", "")
        definition = json_resp.get("definition", "")
        return f"{label}. {definition}" if label or definition else default_seq

    handler = db_handlers.get(db_name)
    if handler is None:
        raise ValueError(f"Unknown database name: {db_name}")

    return handler(identifier)

if __name__ == "__main__":

    base_dir = "data/items"
    for dt in EMBEDDING_DATA_TYPES:
        print(f"Processing {dt}")
        with open(f"{base_dir}/{dt}.txt") as f:
            lines = f.read().splitlines()
        seqs = []

        for line in tqdm(lines):
            db_, id_, count_ = line.split("@")
            seqs.append(get_sequence(id_, db_))
        with open(f"{base_dir}/{dt}_sequences.txt", "w") as f:
            for seq in seqs:
                if "\n" in seq:
                    seq = seq.replace("\n", "\t")
                    print(seq)
                f.write(seq + "\n")
        print(f"Finished processing {dt}")
