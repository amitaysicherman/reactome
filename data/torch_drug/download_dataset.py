import torch
from torchdrug import data, datasets
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--index", type=int, default=1)
args = parser.parse_args()
index = args.index-1

if index == 0:
    datasets.SubcellularLocalization("data/torch_drug/")
elif index == 1:
    datasets.BinaryLocalization("data/torch_drug/")
elif index == 2:
    datasets.GeneOntology("data/torch_drug/")
