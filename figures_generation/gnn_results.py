from common.path_manager import scores_path, figures_path
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_line(line,epoch):
    line = eval(line.replace('nan','None'))
    split = list(line.keys())[0].split('/')[0]
    loss = line[f'{split}/loss']
    all_auc = line[f'{split}/all_auc']
    location_auc = line[f'{split}/fake_location_all']
    protein_auc = line[f'{split}/fake_protein']
    molecule_auc = line[f'{split}/fake_molecule']
    return epoch, split, loss, all_auc, location_auc, protein_auc, molecule_auc


METRICS = ['Loss', 'AUC All', 'AUC Location', 'AUC Protein', 'AUC Molecule']
COLUMNS = ['epoch', 'split', *METRICS]


def parse_fuse_file(file_name):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    data = []
    for i,line in enumerate(lines[1::2]):
        parsed = parse_line(line,i//2)
        if parsed:
            data.append(parsed)
    return pd.DataFrame(data, columns=COLUMNS)


def plot_results(df, title, save_path):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5*len(METRICS), 5))
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for split, group in df.groupby('split'):
            ax.plot(group['epoch'], group[metric], label=split)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
    fig.suptitle(title, fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":

    output_dir = f"{figures_path}/gnn/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fuse_files = glob.glob(f"{scores_path}/gnn*.txt")
    for file in fuse_files:
        df = parse_fuse_file(file)
        title = file.split('/')[-1].split('.')[0].replace("gnn_", "")
        with plt.style.context('tableau-colorblind10', after_reset=True):
            plot_results(df, title, save_path=f"{output_dir}/{title}.png")
