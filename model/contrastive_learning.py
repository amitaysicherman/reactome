import numpy as np

from dataset.fuse_dataset import PairsDataset, SameNameBatchSampler
from dataset.index_manger import NodesIndexManager
import os
from common.data_types import EMBEDDING_DATA_TYPES, PRETRAINED_EMD, DNA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset_builder import get_reactions
from sklearn.metrics import roc_auc_score
from itertools import chain
from common.utils import prepare_files, TYPE_TO_VEC_DIM
from model.models import MultiModalLinearConfig, MiltyModalLinear, EmbModel

EMBEDDING_DATA_TYPES = [x for x in EMBEDDING_DATA_TYPES if x != DNA]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def indexes_to_tensor(indexes, node_index_manager: NodesIndexManager, return_type=True):
    type_ = node_index_manager.index_to_node[indexes[0].item()].type
    array = np.stack([node_index_manager.index_to_node[i.item()].vec for i in indexes])
    if return_type:
        return torch.tensor(array), type_
    return torch.tensor(array)


def save_fuse_model(model: MiltyModalLinear, reconstruction_model: MiltyModalLinear, save_dir, epoch):
    cp_to_remove = []
    for file_name in os.listdir(save_dir):
        if file_name.endswith(".pt"):
            cp_to_remove.append(f"{save_dir}/{file_name}")

    output_file = f"{save_dir}/fuse_{epoch}.pt"
    torch.save(model.state_dict(), output_file)
    # output_file = f"{save_dir}/fuse-recon_{epoch}.pt"
    # torch.save(reconstruction_model.state_dict(), output_file)

    for cp in cp_to_remove:
        os.remove(cp)


def weighted_mean_loss(loss, labels):
    positive_mask = (labels == 1).float().to(device)
    negative_mask = (labels == -1).float().to(device)

    pos_weight = 1.0 / (max(positive_mask.sum(), 1))
    neg_weight = 1.0 / (max(negative_mask.sum(), 1))

    positive_loss = (loss * positive_mask).sum() * pos_weight
    negative_loss = (loss * negative_mask).sum() * neg_weight

    return (positive_loss + negative_loss) / (pos_weight + neg_weight)


def run_epoch(model, reconstruction_model, optimizer, reconstruction_optimizer, loader, contrastive_loss, epoch, recon,
              output_file, part="train", all_to_one=False, use_pretrain=True):
    if len(loader) == 0:
        return 0
    if all_to_one == "inv":
        inv_epoch = EMBEDDING_DATA_TYPES[epoch % len(EMBEDDING_DATA_TYPES)]
    first_epoch = epoch == 0
    is_train = part == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_recon_loss = 0
    all_labels = []
    all_preds = []

    for i, (idx1, idx2, label) in enumerate(loader):
        if not use_pretrain:
            data_1 = idx1.to(device)
            data_2 = idx2.to(device)
            _, type_1 = indexes_to_tensor(idx1, node_index_manager)
            _, type_2 = indexes_to_tensor(idx2, node_index_manager)

        else:
            data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
            data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)
            data_1 = data_1.to(device).float()
            data_2 = data_2.to(device).float()

        if all_to_one:

            if all_to_one == "inv":

                if inv_epoch != type_1 or (first_epoch and inv_epoch == type_1 and inv_epoch == type_2):
                    continue
                if first_epoch:
                    if use_pretrain:
                        out2 = data_2
                    else:
                        out2 = model(data_2, type_2).detach()
                else:
                    out2 = model(data_2, type_2).detach()

            else:
                if not model.have_type((type_1, type_2)):
                    continue
                out2 = data_2

            model_type = type_1 if all_to_one == "inv" else (type_1, type_2)
            out1 = model(data_1, model_type)
            if reconstruction_model is not None:
                recon_1 = reconstruction_model(out1, model_type)
            else:
                recon_1 = data_1
            recon_2 = data_2
        else:
            out1 = model(data_1, type_1)
            out2 = model(data_2, type_2)
            if recon:
                recon_1 = reconstruction_model(out1, type_1)
                recon_2 = reconstruction_model(out2, type_2)

        all_labels.extend((label == 1).cpu().detach().numpy().astype(int).tolist())
        all_preds.extend((0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist())

        cont_loss = contrastive_loss(out1, out2, label.to(device))
        total_loss += cont_loss.mean().item()

        if not is_train:
            continue

        if recon and i % 2 == 1:
            recon_loss = F.mse_loss(recon_1, data_1) + F.mse_loss(recon_2, data_2)
            total_recon_loss += recon_loss.item()
            recon_loss.backward()
            reconstruction_optimizer.step()
            reconstruction_optimizer.zero_grad()

        else:
            cont_loss = weighted_mean_loss(cont_loss, label)
            cont_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    auc = roc_auc_score(all_labels, all_preds)

    msg = f"Epoch {epoch} {part} AUC {auc:.3f} (cont: {total_loss / len(loader):.3f}, " \
          f"recon: {total_recon_loss / len(loader):.3f})"
    if all_to_one == "inv":
        msg += f" {inv_epoch}"
    with open(output_file, "a") as f:
        f.write(msg + "\n")

    print(msg)
    return auc


def get_loader(reactions, node_index_manager, batch_size, split, debug=False):
    dataset = PairsDataset(reactions, node_index_manager, split=split, test_mode=debug)
    sampler = SameNameBatchSampler(dataset, batch_size)
    return DataLoader(dataset, batch_sampler=sampler)


def build_no_pretrained_model(node_index_manager: NodesIndexManager, output_dim: int):
    return EmbModel(len(node_index_manager.index_to_node), output_dim).to(device)


def build_models(fuse_all_to_one, fuse_output_dim, fuse_n_layers, fuse_hidden_dim, fuse_dropout, save_dir):
    if fuse_all_to_one == "" or fuse_all_to_one == "inv":
        names = EMBEDDING_DATA_TYPES
        src_dims = [TYPE_TO_VEC_DIM[x] for x in EMBEDDING_DATA_TYPES]
        dst_dim = [fuse_output_dim] * len(EMBEDDING_DATA_TYPES)

    else:
        names = []
        src_dims = []
        dst_dim = []
        for src in EMBEDDING_DATA_TYPES:
            for dst in EMBEDDING_DATA_TYPES:
                if fuse_all_to_one == "all" or dst == fuse_all_to_one:
                    src_dims.append(TYPE_TO_VEC_DIM[src])
                    names.append((src, dst))
                    dst_dim.append(TYPE_TO_VEC_DIM[dst])

    model_config = MultiModalLinearConfig(
        embedding_dim=src_dims,
        n_layers=fuse_n_layers,
        names=names,
        hidden_dim=fuse_hidden_dim,
        output_dim=dst_dim,
        dropout=fuse_dropout,
        normalize_last=1
    )

    model = MiltyModalLinear(model_config).to(device)

    model_config.save_to_file(f"{save_dir}/config.txt")

    recons_config = MultiModalLinearConfig(
        embedding_dim=dst_dim,
        n_layers=args.fuse_n_layers,
        names=names,
        hidden_dim=args.fuse_hidden_dim,
        output_dim=src_dims,
        dropout=args.fuse_dropout,
        normalize_last=0
    )
    recons_config.save_to_file(f"{save_dir}/config-recon.txt")
    reconstruction_model = MiltyModalLinear(recons_config).to(device)
    return model, reconstruction_model


if __name__ == '__main__':

    from common.args_manager import get_args

    args = get_args()
    save_dir, scores_file = prepare_files(f'fuse2_{args.fuse_name}', skip_if_exists=args.skip_if_exists)

    if args.debug:
        args.fuse_batch_size = 2
    node_index_manager = NodesIndexManager(PRETRAINED_EMD)
    train_reactions, validation_reactions, test_reaction = get_reactions(filter_untrain=not args.fuse_pretrained_start,
                                                                         filter_dna=True,
                                                                         # filter_no_act=True,
                                                                         sample_count=args.gnn_sample)

    if args.fuse_train_all:
        train_reactions = train_reactions + validation_reactions + test_reaction
        validation_reactions = []
        test_reaction = []
    train_loader = get_loader(train_reactions, node_index_manager, args.fuse_batch_size, "train", debug=args.debug)
    valid_loader = get_loader(validation_reactions, node_index_manager, args.fuse_batch_size, "valid", debug=args.debug)
    test_loader = get_loader(test_reaction, node_index_manager, args.fuse_batch_size, "test", debug=args.debug)

    if args.fuse_pretrained_start:
        model, reconstruction_model = build_models(args.fuse_all_to_one, args.fuse_output_dim, args.fuse_n_layers,
                                                   args.fuse_hidden_dim, args.fuse_dropout, save_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fuse_lr)
        reconstruction_optimizer = torch.optim.Adam(chain(model.parameters(), reconstruction_model.parameters()),
                                                    lr=args.fuse_lr)

    else:
        assert args.fuse_all_to_one == "" or args.fuse_all_to_one == "inv"
        assert args.fuse_recon == 0
        model = build_no_pretrained_model(node_index_manager, args.fuse_output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fuse_lr)
        reconstruction_model = None
        reconstruction_optimizer = None
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")
    contrastive_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')
    best_valid_auc = 0

    running_args = {"model": model, "reconstruction_model": reconstruction_model, "optimizer": optimizer,
                    "reconstruction_optimizer": reconstruction_optimizer, "contrastive_loss": contrastive_loss,
                    "recon": args.fuse_recon, "output_file": scores_file, "all_to_one": args.fuse_all_to_one,
                    "use_pretrain": args.fuse_pretrained_start}

    for epoch in range(args.fuse_epochs):
        running_args["epoch"] = epoch
        train_auc = run_epoch(**running_args, loader=train_loader, part="train")
        with torch.no_grad():
            valid_auc = run_epoch(**running_args, loader=valid_loader, part="valid")
            test_auc = run_epoch(**running_args, loader=test_loader, part="test")

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            save_fuse_model(model, reconstruction_model, save_dir, epoch)
