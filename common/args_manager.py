import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="default")

    parser.add_argument("--fuse_batch_size", type=int, default=8192)
    parser.add_argument("--fuse_proteins_molecules_only", type=int, default=0)
    parser.add_argument("--fuse_output_dim", type=int, default=1024)
    parser.add_argument("--fuse_dropout", type=float, default=0.0)
    parser.add_argument("--fuse_lr", type=float, default=1e-3)
    parser.add_argument("--fuse_n_layers", type=int, default=1)
    parser.add_argument("--fuse_hidden_dim", type=int, default=512)
    parser.add_argument("--fuse_recon", type=int, default=0)
    parser.add_argument("--fuse_all_to_one", type=int, default=0)
    parser.add_argument("--fuse_epochs", type=int, default=15)

    parser.add_argument("--gnn_learned_embedding_dim", type=int, default=256)
    parser.add_argument("--gnn_hidden_channels", type=int, default=256)
    parser.add_argument("--gnn_lr", type=float, default=0.001)
    parser.add_argument("--gnn_num_layers", type=int, default=3)
    parser.add_argument("--gnn_conv_type", type=str, default="SAGEConv", choices=["SAGEConv", "TransformerConv"])
    parser.add_argument("--gnn_pretrained_method", type=int, default=1)
    parser.add_argument("--gnn_train_all_emd", type=int, default=0)
    parser.add_argument("--gnn_sample", type=int, default=0)
    parser.add_argument("--gnn_fake_task", type=int, default=1)
    parser.add_argument("--gnn_epochs", type=int, default=15)

    parser.add_argument("--eval_n", type=int, default=10)

    args = parser.parse_args()
    return args
