import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--skip_if_exists", type=int, default=0)
    parser.add_argument("--data_aug", type=str, default="protein", choices=["all", "location", "protein", "molecule"])
    parser.add_argument("--debug", type=int, default=0)

    parser.add_argument("--fuse_batch_size", type=int, default=8192)
    parser.add_argument("--fuse_output_dim", type=int, default=1024)
    parser.add_argument("--fuse_dropout", type=float, default=0.3)
    parser.add_argument("--fuse_lr", type=float, default=1e-3)
    parser.add_argument("--fuse_n_layers", type=int, default=1)
    parser.add_argument("--fuse_hidden_dim", type=int, default=512)
    parser.add_argument("--fuse_recon", type=int, default=0)
    parser.add_argument("--fuse_all_to_one", type=str, default="")
    parser.add_argument("--fuse_epochs", type=int, default=50)
    parser.add_argument("--fuse_name", type=str, default="")
    parser.add_argument("--fuse_pretrained_start", type=int, default=1)

    parser.add_argument("--gnn_learned_embedding_dim", type=int, default=256)
    parser.add_argument("--gnn_hidden_channels", type=int, default=256)
    parser.add_argument("--gnn_lr", type=float, default=1e-3)
    parser.add_argument("--gnn_num_layers", type=int, default=3)
    parser.add_argument("--gnn_conv_type", type=str, default="SAGEConv", choices=["SAGEConv", "TransformerConv"])
    parser.add_argument("--gnn_pretrained_method", type=int, default=1)
    parser.add_argument("--gnn_train_all_emd", type=int, default=0)
    parser.add_argument("--gnn_sample", type=int, default=0)
    parser.add_argument("--gnn_fake_task", type=int, default=1)
    parser.add_argument("--gnn_epochs", type=int, default=50)
    parser.add_argument("--gnn_fuse_name", type=str, default="default")
    parser.add_argument("--gnn_last_or_concat", type=int, default=0)
    parser.add_argument("--gnn_drop_out", type=float, default=0.3)
    # parser.add_argument("--gnn_reaction_or_mean", type=int, default=0)

    parser.add_argument("--eval_n", type=int, default=10)

    parser.add_argument("--seq_use_trans", type=int, default=0)
    parser.add_argument("--seq_size", type=str, default="s", choices=['s', 'm', 'l'])

    parser.add_argument("--seq_a", type=float, default=[0.15, 0.15, 0], nargs='+')
    parser.add_argument("--seq_k", type=int, default=16)
    parser.add_argument("--seq_aug_factor", type=int, default=10)
    parser.add_argument("--all_to_prot", type=int, default=1)

    args = parser.parse_args()
    if args.fuse_name == "" or args.fuse_name == "0":
        args.fuse_name = args.name
    return args
