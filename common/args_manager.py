import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # preproccecing args

    parser.add_argument("--self_token", type=str, default="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")
    parser.add_argument("--protein_emd", type=str, default="ProtBertT5-xl")
    parser.add_argument("--mol_emd", type=str, default="pebchem10m")
    parser.add_argument("--prep_reactome_dtype", type=str, default="all",
                        choices=["all", "protein", "molecule", "text", 'dna', "label"])
    parser.add_argument("--task_name", type=str, default="BACE")
    parser.add_argument("--task_suffix", type=str, default="")
    parser.add_argument("--downstream_task", type=str, default="pd", choices=["pd", "loc", "go", "rrf", "cl", "mol"])

    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--skip_if_exists", type=int, default=0)
    parser.add_argument("--data_aug", type=str, default="protein", choices=["all", "location", "protein", "molecule"])
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--max_no_improve", type=int, default=15)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--fuse_batch_size", type=int, default=8192)
    parser.add_argument("--fuse_output_dim", type=int, default=1024)
    parser.add_argument("--fuse_dropout", type=float, default=0.3)
    parser.add_argument("--fuse_lr", type=float, default=1e-3)
    parser.add_argument("--fuse_n_layers", type=int, default=1)
    parser.add_argument("--fuse_hidden_dim", type=int, default=512)
    parser.add_argument("--fuse_recon", type=int, default=0)
    parser.add_argument("--fuse_all_to_one", type=str, default="protein")
    parser.add_argument("--fuse_self_move", type=int, default=1)
    parser.add_argument("--fuse_train_all", type=int, default=1)
    parser.add_argument("--fuse_max_no_improve", type=int, default=5)
    parser.add_argument("--fuse_triples", type=int, default=1)

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

    parser.add_argument("--seq_use_trans", type=int, default=1)
    parser.add_argument("--seq_size", type=str, default="s", choices=['s', 'm', 'l'])

    parser.add_argument("--seq_a", type=float, default=[0.15, 0.15, 0], nargs='+')
    parser.add_argument("--seq_k", type=int, default=16)
    parser.add_argument("--seq_aug_factor", type=int, default=10)
    parser.add_argument("--all_to_prot", type=int, default=1)

    # drug protein args
    parser.add_argument("--dp_fuse_base", type=str, default="")
    parser.add_argument("--dp_m_fuse", type=int, default=1)
    parser.add_argument("--dp_p_fuse", type=int, default=1)
    parser.add_argument("--dp_m_model", type=int, default=0)
    parser.add_argument("--dp_p_model", type=int, default=0)
    parser.add_argument("--dp_only_rand", type=int, default=0)
    parser.add_argument("--dp_fuse_freeze", type=int, default=1)
    parser.add_argument("--dp_bs", type=int, default=64)
    parser.add_argument("--dp_lr", type=float, default=5e-5)
    parser.add_argument("--dp_n_layers", type=int, default=1)
    parser.add_argument("--dp_hidden_dim", type=int, default=-1)
    parser.add_argument("--dp_drop_out", type=float, default=0.0)
    parser.add_argument("--dp_print", type=int, default=1)
    parser.add_argument("--db_dataset", type=str, default="DrugBank", choices=["Davis", "DrugBank", "KIBA", "human"])

    # cafa args
    parser.add_argument("--cafa_use_fuse", type=int, default=1)
    parser.add_argument("--cafa_use_model", type=int, default=1)
    parser.add_argument("--cafa_fuse_freeze", type=int, default=1)
    parser.add_argument("--cafa_task", type=str, default="mf", choices=["mf", "bp", "cc"])

    # localization args
    parser.add_argument("--loc_bin", type=int, default=0)

    # go args
    parser.add_argument("--go_task", type=str, default="MF", choices=["MF", "BP", "CC"])

    # mol
    parser.add_argument("--mol_task", type=str, default="BACE", choices=["BACE", "BBBP", "ClinTox", "HIV", "SIDER"])

    # add tasks:
    parser.add_argument("--task_output_prefix", type=str, default="")

    args = parser.parse_args()
    if args.dp_fuse_base == "":
        args.dp_fuse_base = f'{args.name}'
    if args.fuse_all_to_one == "0":
        args.fuse_all_to_one = ""
    if args.fuse_name == "" or args.fuse_name == "0":
        args.fuse_name = args.name
    return args
