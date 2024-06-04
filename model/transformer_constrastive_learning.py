import numpy as np
from dataset.index_manger import NodesIndexManager
import os
from common.utils import reaction_from_str, TYPE_TO_VEC_DIM
from common.data_types import Reaction, EMBEDDING_DATA_TYPES, NodeTypes
from itertools import combinations
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataset.dataset_builder import have_unkown_nodes, have_dna_nodes
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from sklearn.metrics import roc_auc_score
from itertools import chain
from common.path_manager import reactions_file, item_path, model_path, scores_path
from model.models import MultiModalLinearConfig, MiltyModalLinear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
TEST_MODE = False


def vecs_from_reaction(reaction: Reaction, nodes_index_manager: NodesIndexManager, proteins_molecules_only: bool):
    elements = []
    reaction_elements = reaction.inputs + reaction.outputs + sum([x.entities for x in reaction.catalysis], [])
    for reaction_element in reaction_elements:
        node = nodes_index_manager.name_to_node[reaction_element.get_db_identifier()]
        if proteins_molecules_only and (node.type != NodeTypes.protein or node.type != NodeTypes.molecule):
            continue
        elements.append(node.vec)
    if not proteins_molecules_only:
        for mod in sum([list(x.modifications) for x in reaction_elements], []):
            node = nodes_index_manager.name_to_node["TEXT@" + mod]
            elements.append(node.vec)
        for act in [x.activity for x in reaction.catalysis]:
            node = nodes_index_manager.name_to_node["GO@" + act]
            elements.append(node.vec)
    return elements


class PairsDataset(Dataset):
    def __init__(self, nodes_index_manager: NodesIndexManager, proteins_molecules_only: bool, split="train"):
        self.nodes_index_manager = nodes_index_manager

        with open(reactions_file) as f:
            lines = f.readlines()
        lines = sorted(lines, key=lambda x: reaction_from_str(x).date)
        reactions = [reaction_from_str(line) for line in lines]
        if split == "train":
            reactions = reactions[:int(len(reactions) * 0.8)]
        else:
            reactions = reactions[int(len(reactions) * 0.8):]
        reactions = [reaction for reaction in reactions if
                     not have_unkown_nodes(reaction, nodes_index_manager, check_output=True)]
        if proteins_molecules_only:
            reactions = [reaction for reaction in reactions if
                         not have_dna_nodes(reaction, nodes_index_manager, check_output=True)]
        self.reactions_vecs = []
        for reaction in tqdm(reactions):
            self.reactions_vecs.append(vecs_from_reaction(reaction, nodes_index_manager, proteins_molecules_only))

        self.data = []
        for i in tqdm(range(len(self.reactions_vecs))):
            pos_vecs = self.reactions_vecs[i]
            neg_vecs = random.choice(self.reactions_vecs)
            self.data.append((pos_vecs, neg_vecs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def remove_vecs_files(run_name):
    if not os.path.exists(f"{item_path}/{run_name}"):
        return
    for file_name in os.listdir(f"{item_path}/{run_name}"):
        if file_name.endswith(".npy"):
            os.remove(f"{item_path}/{run_name}/{file_name}")


def save_fuse_model(model: MiltyModalLinear, reconstruction_model: MiltyModalLinear, save_dir, epoch):
    output_file = f"{save_dir}/fuse_{epoch}.pt"
    torch.save(model.state_dict(), output_file)
    output_file = f"{save_dir}/fuse-recon_{epoch}.pt"
    torch.save(reconstruction_model.state_dict(), output_file)


def weighted_mean_loss(loss, labels):
    positive_mask = (labels == 1).float().to(device)
    negative_mask = (labels == -1).float().to(device)

    pos_weight = 1.0 / (max(positive_mask.sum(), 1))
    neg_weight = 1.0 / (max(negative_mask.sum(), 1))

    positive_loss = (loss * positive_mask).sum() * pos_weight
    negative_loss = (loss * negative_mask).sum() * neg_weight

    return (positive_loss + negative_loss) / (pos_weight + neg_weight)


def run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, loader, contrastive_loss, epoch, recon,
              output_file, is_train=True, all_to_prot=False):
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_recon_loss = 0
    all_labels = []
    all_preds = []

    for i, (idx1, idx2, label) in enumerate(loader):
        data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
        data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)

        data_1 = data_1.to(device).float()
        data_2 = data_2.to(device).float()

        if all_to_prot:
            if type_1 == NodeTypes.protein:
                out1 = data_1
                out2 = model(data_2, type_2)
                recon_2 = reconstruction_model(out2, type_2)
                recon_loss = F.mse_loss(recon_2, data_2)
            else:  # type_2 == NodeTypes.protein
                out2 = data_2
                out1 = model(data_1, type_1)
                recon_1 = reconstruction_model(out1, type_1)
                recon_loss = F.mse_loss(recon_1, data_1)
        else:
            out1 = model(data_1, type_1)
            recon_1 = reconstruction_model(out1, type_1)
            out2 = model(data_2, type_2)
            recon_2 = reconstruction_model(out2, type_2)
            recon_loss = F.mse_loss(recon_1, data_1) + F.mse_loss(recon_2, data_2)

        cont_loss = contrastive_loss(out1, out2, label.to(device))

        all_labels.extend((label == 1).cpu().detach().numpy().astype(int).tolist())
        all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist())

        total_loss += cont_loss.mean().item()

        total_recon_loss += recon_loss.item()
        if not is_train:
            continue
        if not recon or i % 2 == 0:
            cont_loss = weighted_mean_loss(cont_loss, label)
            cont_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            recon_loss.backward()
            reconstruction_optimizer.step()
            reconstruction_optimizer.zero_grad()
    auc = roc_auc_score(all_labels, all_preds)

    msg = f"Epoch {epoch} {'Train' if is_train else 'Test'} AUC {auc:.3f} (cont: {total_loss / len(loader):.3f}, " \
          f"recon: {total_recon_loss / len(loader):.3f})"
    with open(output_file, "a") as f:
        f.write(msg + "\n")
    print(msg)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--proteins_molecules_only", type=int, default=0)
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--recon", type=int, default=0)
    parser.add_argument("--name", type=str, default="all_to_port")
    parser.add_argument("--all_to_prot", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.all_to_prot:
        args.output_dim = TYPE_TO_VEC_DIM[NodeTypes.protein]
    EPOCHS = args.epochs

    run_name = args.name
    if not TEST_MODE:
        remove_vecs_files(run_name)

    if args.proteins_molecules_only:
        EMBEDDING_DATA_TYPES = [NodeTypes.protein, NodeTypes.molecule]
    if TEST_MODE:
        args.batch_size = 2
    node_index_manager = NodesIndexManager()
    dataset = PairsDataset(node_index_manager, proteins_molecules_only=args.proteins_molecules_only,
                           all_to_prot=args.all_to_prot)

    sampler = SameNameBatchSampler(dataset, args.batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler)

    test_dataset = PairsDataset(node_index_manager, split="test",
                                proteins_molecules_only=args.proteins_molecules_only, all_to_prot=args.all_to_prot)
    test_sampler = SameNameBatchSampler(test_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    if args.proteins_molecules_only:
        emb_dim = {NodeTypes.protein: 1024, NodeTypes.molecule: 768}
    else:
        emb_dim = {NodeTypes.protein: 1024, NodeTypes.molecule: 768, NodeTypes.dna: 768, NodeTypes.text: 768}

    save_dir = f"{model_path}/fuse_{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file_name in os.listdir(save_dir):
        if file_name.endswith(".pt"):
            os.remove(f"{save_dir}/{file_name}")
    scores_file = f"{scores_path}/fuse_{run_name}.txt"
    if os.path.exists(scores_file):
        os.remove(scores_file)
    model_config = MultiModalLinearConfig(
        embedding_dim=list(emb_dim.values()),
        n_layers=args.n_layers,
        names=list(emb_dim.keys()),
        hidden_dim=args.hidden_dim,
        output_dim=[args.output_dim] * len(emb_dim),
        dropout=args.dropout,
        normalize_last=1
    )

    model = MiltyModalLinear(model_config).to(device)

    model_config.save_to_file(f"{save_dir}/config.txt")

    recons_config = MultiModalLinearConfig(
        embedding_dim=[args.output_dim] * len(emb_dim),
        n_layers=args.n_layers,
        names=list(emb_dim.keys()),
        hidden_dim=args.hidden_dim,
        output_dim=list(emb_dim.values()),
        dropout=args.dropout,
        normalize_last=0
    )
    recons_config.save_to_file(f"{save_dir}/config-recon.txt")

    reconstruction_model = MiltyModalLinear(recons_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reconstruction_optimizer = torch.optim.Adam(chain(model.parameters(), reconstruction_model.parameters()),
                                                lr=args.lr)
    contrastive_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    for epoch in range(EPOCHS):
        run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, loader, contrastive_loss, epoch,
                  args.recon, scores_file, is_train=True, all_to_prot=args.all_to_prot)

        run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, test_loader, contrastive_loss,
                  epoch, args.recon, scores_file, is_train=False, all_to_prot=args.all_to_prot)

        save_fuse_model(model, reconstruction_model, save_dir, epoch)
