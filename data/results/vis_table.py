import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_type = "other"  # "fuse_configs"

best_fuse_confs = [
    [{'fuse_o_dim': 1024, 'fuse_recon': 1, 'fuse_mp': 0}, {'fuse_o_dim': 256, 'fuse_recon': 1, 'fuse_mp': 1}]]
data = pd.read_csv('scores.csv')
data = data.sort_values(by=['train_all_emd', 'pretrained_method', 'fuse_o_dim', 'fuse_recon', 'fuse_mp'])
metrics = ['protein_protein', 'molecule_molecule', 'protein_both', 'molecule_both']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
ax = axes.ravel()


def column_to_label(row):
    row = {x: y for x, y in row.items() if 'mean' not in x and 'std' not in x}
    size = "Large" if row['fuse_o_dim'] == 1024 else "Small"
    recons = "V" if row['fuse_recon'] == 1 else "X"
    proteins_molecules = "Proteins & Molecules" if row['fuse_mp'] == 1 else "All"
    fuse_name = f'Size:{size}\nReconstruction:{recons}\nData Types:{proteins_molecules}'

    is_train = "(trained)" if row['train_all_emd'] == 1 else "(freezed)"

    if row['pretrained_method'] == 0:
        name = f"No pretrained {is_train}"
    elif row['pretrained_method'] == 1:
        name = f"Pretrained {is_train}"
    elif row['pretrained_method'] == 2:
        name = f"Pretrained Fuse {is_train}\n{fuse_name}"
    else:  # row['pretrained_method']==3:
        name = f"Pretrained Concat {is_train}\n{fuse_name}"
    return name


for i, metric in enumerate(metrics):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Calculate number of bars and bar positions

    # Create bar labels by concatenating hyperparameter values
    bar_labels = []
    for _, row in data.iterrows():
        bar_labels.append(column_to_label(row))
    labels = np.array(bar_labels)
    means = data[mean_col] * 100
    stds = data[std_col] * 100
    if plot_type == "fuse_configs":
        mask = np.array(["Fuse" in l and "freezed" in l for l in bar_labels])
        bar_labels = [x.replace("Pretrained Fuse (freezed)\n", "") for x in bar_labels]
        labels = np.array(bar_labels)

    elif plot_type=="other":
        mask = ~np.array(["Fuse" in l and "freezed" in l for l in bar_labels])

    labels = labels[mask]
    means = means[mask]
    stds = stds[mask]

    max_std = stds.max()
    x_min = means.min() - 2 * max_std
    x_max = means.max() + 2 * max_std

    num_bars = len(labels)
    bar_positions = range(len(labels))

    bars = ax[i].barh(bar_positions, means, xerr=stds, capsize=3, alpha=0.7)
    ax[i].set_title(f"{metric.replace('_', ' ').title()} Scores")
    ax[i].set_yticks(bar_positions)
    ax[i].set_yticklabels(labels)
    ax[i].set_xlim(x_min, x_max)  # Set x-axis limit
    for bar, std in zip(bars, stds):
        bar_width = bar.get_width()
        annotation_x = bar_width + std  # Position annotation at std distance + 5 from bar end
        ax[i].annotate(f"{bar_width:.1f}", xy=(annotation_x, bar.get_y() + bar.get_height() / 2),
                       xytext=(0, 0), textcoords="offset points", ha='left', va='center')

    fig.tight_layout()

plt.savefig(f"{plot_type}.png", dpi=300)
plt.show()
