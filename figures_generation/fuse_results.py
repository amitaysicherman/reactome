from common.path_manager import scores_path,figures_path
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

def parse_line(line):
    pattern = r'Epoch (\d+) (Test|Train) AUC (\d+\.\d+) \(cont: (\d+\.\d+), recon: (\d+\.\d+)\)'
    match = re.match(pattern, line)
    if match:
        epoch = int(match.group(1))
        split = match.group(2)
        auc = float(match.group(3))
        cont = float(match.group(4))
        recon = float(match.group(5))
        return epoch, split, auc, cont, recon
    else:
        return None


METRICS = ['AUC', 'Contrastive Loss', 'Reconstruction Loss']
COLUMNS = ['epoch', 'split', *METRICS]


def parse_fuse_file(file_name):
    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    data = []
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            data.append(parsed)
    return pd.DataFrame(data, columns=COLUMNS)


def plot_results(df, title,save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
    output_dir=f"{figures_path}/fuse/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fuse_files = glob.glob(f"{scores_path}/fuse*.txt")
    for file in fuse_files:
        df = parse_fuse_file(file)
        title = file.split('/')[-1].split('.')[0].replace("fuse_", "")
        with plt.style.context('tableau-colorblind10', after_reset=True):
            plot_results(df, title,save_path=f"{output_dir}/{title}.png")
