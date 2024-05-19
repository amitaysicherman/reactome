import dataclasses

from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv, SAGEConv, Linear
from torch_geometric.nn import DataParallel
from common import DATA_TYPES, EMBEDDING_DATA_TYPES
import numpy as np
from index_manger import NodesIndexManager, get_edges_values, NodeTypes, PRETRAINED_EMD_FUSE, NO_PRETRAINED_EMD, \
    PRETRAINED_EMD
from dataset_builder import ReactionDataset
from tagging import ReactionTag
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import random
from torch_geometric.loader import DataLoader

from sklearn.manifold import TSNE
import wandb
import seaborn as sns
from dataset_builder import REAL, FAKE_LOCATION_ALL, FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE

sns.set()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REACTION = NodeTypes().reaction
wandb_key = "4bc00ebfe22837b81a77a3fe6b27efdac2cde667"
type_to_color = {
    "protein": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    "molecule": (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    "dna": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    "complex": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
    "catalysis_protein": (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    "catalysis_molecule": (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    "catalysis_dna": (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
    "catalysis_activity": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
}

batch_size = 1
EPOCHS = 5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Scorer:
    def __init__(self, name, class_names):
        self.name = name
        self.class_names = class_names
        self.index_to_class = {i: class_name for i, class_name in enumerate(class_names)}
        self.reset()

    def reset(self):
        self.count = 0
        self.batch_count = 0
        self.loss = 0
        self.index_to_class = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.true_probs = {class_name: [] for class_name in self.class_names}  # New for true probs
        self.pred_probs = {class_name: [] for class_name in self.class_names}  # New for pred probs
        for class_name in self.class_names:
            setattr(self, f'{class_name}_auc', 0)

    def add(self, y, pred, out, loss, class_names=None):
        self.count += len(y)
        self.batch_count += 1
        self.loss += loss
        if class_names:
            for i in range(len(y)):
                class_name = class_names[i]
                self.true_probs[class_name].extend(y[i].tolist())
                self.pred_probs[class_name].extend(sigmoid(out[i]).tolist())
        else:
            for i, class_name in self.index_to_class.items():
                self.true_probs[class_name].extend(y[:, i].tolist())
                self.pred_probs[class_name].extend(sigmoid(out[:, i]).tolist())

    def compute_auc_per_class(self):
        auc_dict = {}
        all_auc = []
        real_probs = self.true_probs[REAL]
        real_preds = self.pred_probs[REAL]
        for class_name in self.class_names:
            if class_name == REAL:
                continue
            true_probs = self.true_probs[class_name] + real_probs
            pred_probs = self.pred_probs[class_name] + real_preds
            if len(true_probs) > 0 and len(np.unique(true_probs)) > 1:
                auc_dict[class_name] = roc_auc_score(true_probs, pred_probs)
                all_auc.append(auc_dict[class_name])
            else:
                auc_dict[class_name] = float('nan')
        auc_dict["all"] = np.mean(all_auc)
        return auc_dict

    def get_log(self):
        auc_dict = self.compute_auc_per_class()
        metrics = {
            f"{self.name}/loss": self.loss / self.batch_count,
            f'{self.name}/all_auc': auc_dict['all'],
        }
        for class_name in self.class_names:
            if class_name == REAL:
                continue
            name = f"{self.name}/{class_name}"
            metrics[f"{name}"] = auc_dict[class_name]
            # if WB:
            #     ground_truth = self.true_probs[class_name]
            #     predictions_true = np.array(self.pred_probs[class_name])
            #     predictions_false = 1 - predictions_true
            #     predictions = np.stack([predictions_false, predictions_true], axis=1)
            #     if len(np.unique(ground_truth)) == 2:
            #         metrics[f"{name}/roc"] = wandb.plot.roc_curve(np.array(ground_truth).astype(int), predictions,
            #                                                       labels=["no " + name, name], classes_to_plot=[1])
        self.reset()
        return metrics


class PartialFixedEmbedding(nn.Module):
    def __init__(self, node_index_manager: NodesIndexManager, learned_embedding_dim, train_all_emd):
        super(PartialFixedEmbedding, self).__init__()
        self.node_index_manager = node_index_manager
        self.embeddings = nn.ModuleDict()
        self.embeddings[NodeTypes.reaction] = nn.Embedding(1, learned_embedding_dim).to(device)
        self.embeddings[NodeTypes.complex] = nn.Embedding(1, learned_embedding_dim).to(device)
        for dtype in DATA_TYPES:
            if dtype in EMBEDDING_DATA_TYPES:
                dtype_indexes = range(node_index_manager.dtype_to_first_index[dtype],
                                      node_index_manager.dtype_to_last_index[dtype])
                vectors = np.stack([node_index_manager.index_to_node[i].vec for i in dtype_indexes])
                self.embeddings[dtype] = nn.Embedding.from_pretrained(
                    torch.tensor(vectors, dtype=torch.float32).to(device))
                if not train_all_emd:
                    self.embeddings[dtype].requires_grad_(False)
            else:
                n_samples = node_index_manager.dtype_to_last_index[dtype] - node_index_manager.dtype_to_first_index[
                    dtype]
                self.embeddings[dtype] = nn.Embedding(n_samples, learned_embedding_dim).to(device)

    def forward(self, x):
        x = x.squeeze(1)
        dtype = self.node_index_manager.index_to_node[x[0].item()].type
        if dtype == NodeTypes.reaction or dtype == NodeTypes.complex:
            return self.embeddings[dtype](torch.zeros_like(x).to(device))
        x = x - self.node_index_manager.dtype_to_first_index[dtype]
        x = self.embeddings[dtype](x)
        return x


conv_type_to_class = {
    "TransformerConv": TransformerConv,
    "SAGEConv": SAGEConv
}

conv_type_to_args = {
    "TransformerConv": {},
    "SAGEConv": {"normalize": True}
}


class HeteroGNN(torch.nn.Module):
    def __init__(self, node_index_manager, hidden_channels, out_channels, num_layers, learned_embedding_dim,
                 train_all_emd, save_activation=False, conv_type="SAGEConv", return_reaction_embedding=False):
        super().__init__()
        self.emb = PartialFixedEmbedding(node_index_manager, learned_embedding_dim, train_all_emd).to(device)
        self.convs = torch.nn.ModuleList()
        self.save_activations = defaultdict(list)

        for i in range(num_layers):
            conv_per_edge = {}
            for edge in get_edges_values():
                e_conv = conv_type_to_class[conv_type](-1, hidden_channels, **conv_type_to_args[conv_type]).to(
                    device)
                if save_activation:
                    if edge[2] == NodeTypes.reaction or edge[2] == NodeTypes.complex:
                        e_conv.register_forward_hook(self.get_hook(edge, i))
                conv_per_edge[edge] = e_conv
            conv = HeteroConv(conv_per_edge, aggr='sum')
            self.convs.append(conv)
        self.lin_reaction = Linear(hidden_channels, out_channels).to(device)
        self.return_reaction_embedding = return_reaction_embedding

    def get_hook(self, dtype, level):
        def save_hook(_, __, output):
            saved_data = output.cpu().detach().numpy()
            for i in range(saved_data.shape[0]):
                self.save_activations[(dtype, level)].append(saved_data[i])

        return save_hook

    def forward(self, x_dict, edge_index_dict):
        for key, x in x_dict.items():
            x_dict[key] = self.emb(x)
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.return_reaction_embedding:
            return self.lin_reaction(x_dict[REACTION]), x_dict[REACTION]
        return self.lin_reaction(x_dict[REACTION])

    def plot_activations(self, title="", reduce_dim_method="TSNE", max_samples=500, last_layer_only=True):
        for layer_index in range(len(self.convs)):
            for type_ in [NodeTypes.reaction, NodeTypes.complex]:
                if last_layer_only and layer_index != len(self.convs) - 1:
                    continue
                combined_data = []
                labels = []
                for (key, level), values in self.save_activations.items():
                    if level != layer_index:
                        continue
                    if key[2] != type_:
                        continue
                    if len(values) > max_samples:
                        values = random.sample(values, max_samples)
                    combined_data.append(values)
                    labels.append(key[1].split("_to")[0])
                flat_data = sum(combined_data, [])
                if not len(flat_data):
                    continue
                combined_data_stack = np.stack(flat_data)
                if reduce_dim_method == "PCA":
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(combined_data_stack)
                else:
                    tsne = TSNE(n_components=2, random_state=42)
                    reduced_data = tsne.fit_transform(combined_data_stack)
                combined_data_reduced = []
                start_index = 0
                for d in combined_data:
                    combined_data_reduced.append(reduced_data[start_index:start_index + len(d)])
                    start_index += len(d)
                plt.figure(figsize=(6, 6))
                for i, data in enumerate(combined_data_reduced):
                    plt.scatter(data[:, 0], data[:, 1], label=labels[i], c=type_to_color[labels[i]], s=10)
                plt.legend(loc='upper left')
                params = dict(axis='both', which='both', bottom=False, top=False, labelbottom=False)
                plt.tick_params(**params)
                plt.title(f"{title} {type_} {layer_index} {reduce_dim_method}")

                plt.tight_layout()
                plt.savefig(f"data/fig/{title}_{type_}_{layer_index}_{reduce_dim_method}.png", dpi=300)
                plt.show()

        self.save_activations = defaultdict(list)


def get_data(root="data/items", sample=0, location_augmentation_factor=2, entity_augmentation_factor=1,
             train_test_split=0.8,
             split_method="date"):
    if FAKE_TASK:

        dataset = ReactionDataset(root=root, sample=sample, location_augmentation_factor=location_augmentation_factor,
                                  # replace_entity_location=location_augmentation_factor,
                                  # molecule_similier_factor=entity_augmentation_factor,
                                  molecule_random_factor=entity_augmentation_factor,
                                  # protein_similier_factor=entity_augmentation_factor,
                                  protein_random_factor=entity_augmentation_factor, order=split_method).reactions
    else:
        dataset = ReactionDataset(root=root, sample=sample, order=split_method).reactions
    print(len(dataset))
    tags = torch.stack([reaction.tags for reaction in dataset])
    pos_classes_weights = (1 - tags.mean(dim=0)) / tags.mean(dim=0)
    if FAKE_TASK:
        pos_classes_weights = pos_classes_weights[-1]
    else:
        pos_classes_weights = pos_classes_weights[:-1]
    train_dataset = dataset[:int(len(dataset) * train_test_split)]
    test_dataset = dataset[int(len(dataset) * train_test_split):]
    return train_dataset, test_dataset, pos_classes_weights.to(device)


def get_models(learned_embedding_dim, hidden_channels, out_channels, num_layers, train_all_emd, lr, pretrained_method,
               layer_type="SAGEConv"):
    # emb_model = PartialFixedEmbedding(node_index_manager, learned_embedding_dim, train_all_emd).to(device)
    node_index_manager = NodesIndexManager(fuse_vec=pretrained_method)
    model = HeteroGNN(node_index_manager, hidden_channels=hidden_channels, out_channels=out_channels,
                      num_layers=num_layers,
                      learned_embedding_dim=learned_embedding_dim, train_all_emd=train_all_emd,
                      conv_type=layer_type).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # if not lr_emb:
    #     lr_emb = lr
    # emb_optimizer = torch.optim.Adam(emb_model.parameters(), lr=lr_emb)
    return model, optimizer


def run_model(data, model, optimizer, scorer: Scorer, is_train=True):
    x_dict = {key: data.x_dict[key].to(device) for key in data.x_dict.keys()}
    y = data['tags'].to(device)
    augmentation_types = data['augmentation_type']
    y = y.float()
    if FAKE_TASK:
        y = y[-1:]
    else:
        y = y[:-1]

    edge_index_dict = {key: data.edge_index_dict[key].to(device) for key in data.edge_index_dict.keys()}
    out = model(x_dict, edge_index_dict)
    pred = (out > 0).to(torch.int32)
    y = y.reshape(out.shape)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_classes_weights)(out, y)
    scorer.add(y.cpu().numpy(), pred.detach().cpu().numpy(), out.detach().cpu().numpy(), loss.item(),
               class_names=augmentation_types if FAKE_TASK else None)
    if is_train:
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()


def train(model, optimizer, batch_size, log_func, epochs=EPOCHS, save_model=""):
    for i in range(epochs):

        train_score = Scorer("train", scores_tag_names)
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for data_index, data in tqdm(enumerate(train_data)):
            run_model(data, model, optimizer, train_score)
        log_func(train_score.get_log(), i)
        test_score = Scorer("test", scores_tag_names)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for data_index, data in tqdm(enumerate(test_data)):
            run_model(data, model, optimizer, test_score, False)
        log_func(test_score.get_log(), i)
        if save_model:
            if FAKE_TASK:
                name = f'data/model/model_fake_{save_model}_{i}.pt'
            else:
                name = f'data/model/model_mlc_{save_model}_{i}.pt'
            torch.save(model.state_dict(), name)
            torch.save(optimizer.state_dict(), name.replace("model_", "optimizer_"))


def run_wandb():
    sweep_config = {
        'method': 'grid',
    }

    metric = {
        'name': 'test/all_auc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric
    parameters_dict = {
        'learned_embedding_dim': {
            'values': [256],
            'distribution': 'categorical'
        },
        'hidden_channels': {
            'values': [256],
            'distribution': 'categorical'
        },
        'lr': {
            'values': [1e-3],
            'distribution': 'categorical'
        },
        'layer_type': {
            'values': ["SAGEConv"],
            'distribution': 'categorical'
        },
        'num_layers': {
            'values': [3],
            'distribution': 'categorical'
        },
        'pretrained_method': {
            'values': [PRETRAINED_EMD, PRETRAINED_EMD_FUSE, NO_PRETRAINED_EMD],
            'distribution': 'categorical'
        },
        'train_all_emd': {
            'values': [True, False],
            'distribution': 'categorical'
        }

    }

    sweep_config['parameters'] = parameters_dict

    def wandb_train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            model, optimizer = get_models(config.learned_embedding_dim,
                                          config.hidden_channels,
                                          len(tag_names),
                                          config.num_layers,
                                          config.train_all_emd, config.lr, config.pretrained_method, config.layer_type)
            save_prefix = f"{config.hidden_channels}"
            train(model, optimizer, batch_size,
                  lambda x, step: wandb.log(x, step=step), save_model=save_prefix)

    sweep_id = wandb.sweep(sweep_config, project="reactome-real-rake")
    wandb.agent(sweep_id, wandb_train, count=100)


def run_local():
    learned_embedding_dim = 256
    hidden_channels = 256
    lr = 0.001
    num_layers = 3
    batch_size = 1
    pretrained_method = PRETRAINED_EMD
    train_all_emd = False
    layer_type = "SAGEConv"

    def print_train_log(x, step):
        print(step)
        print(x)
        # if "train/loss" in x:
        #     print(f"Step {step} Loss: {x['train/loss']},AUC: {x['train/all_auc']}")
        # else:
        #     print(f"Step {step} Loss: {x['test/loss']},AUC: {x['test/all_auc']}")

    model, optimizer = get_models(learned_embedding_dim, hidden_channels,
                                  len(tag_names),
                                  num_layers, train_all_emd, lr, pretrained_method, layer_type=layer_type)

    train(model, optimizer, batch_size, print_train_log, save_model=hidden_channels)


def args_to_str(args):
    args_dict = vars(args)
    sorted_args = sorted(args_dict.items())
    args_str = '-'.join([f"{key}_{value}" for key, value in sorted_args])
    return args_str


def run_with_args(args):
    file_name = f"data/scores/{args_to_str(args)}.txt"

    def save_to_file(x, step):
        with open(file_name, "a") as f:
            f.write(f"{step}\n")
            f.write(f"{x}\n")

    model, optimizer = get_models(args.learned_embedding_dim, args.hidden_channels,
                                  len(tag_names),
                                  args.num_layers, args.train_all_emd, args.lr, args.pretrained_method,
                                  layer_type=args.layer_type)

    train(model, optimizer, batch_size, save_to_file, save_model=args.hidden_channels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learned_embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--layer_type", type=str, default="SAGEConv", choices=["SAGEConv", "TransformerConv"])
    parser.add_argument("--pretrained_method", type=str, default=PRETRAINED_EMD,
                        choices=[PRETRAINED_EMD, PRETRAINED_EMD_FUSE, NO_PRETRAINED_EMD])
    parser.add_argument("--train_all_emd", type=bool, default=False, choices=[True, False])
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--fake_task", type=bool, default=True)
    args = parser.parse_args()
    FAKE_TASK = args.fake_task

    if FAKE_TASK:
        tag_names = ["fake"]
        scores_tag_names = [REAL, FAKE_LOCATION_ALL, FAKE_LOCATION_SINGLE, FAKE_PROTEIN, FAKE_MOLECULE]
    else:
        tag_names = [x for x in dataclasses.asdict(ReactionTag()).keys() if x != "fake"]
        scores_tag_names = tag_names

    train_dataset, test_dataset, pos_classes_weights = get_data(sample=args.sample)
    run_with_args(args)
    # if WB:
    #     run_wandb()
    # else:
    #     run_local()
