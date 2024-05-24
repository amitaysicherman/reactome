import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metric_to_name = {
    'protein_protein': "Replace Protein in Protein Only Reaction",
    'molecule_molecule': "Replace Molecule in Molecule Only Reaction",
    'protein_both': "Replace Protein in Complex Reaction",
    'molecule_both': "Replace Molecule in Complex Reaction",
}


with plt.style.context('tableau-colorblind10', after_reset=True):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    data = [7816, 4137, 417]
    recipe = ["Complex (63%)",
              "Protein Only (33%)",
              "Molecules Only (4%)"]

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-135)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title("Reaction Types")
    fig.tight_layout()

    plt.savefig("reactions_type.png", dpi=300)

    plt.show()


file_name_to_figsize = {"fuse_configs": (15, 10), "fuse_methods": (10, 7), "train_vs_freeze": (12, 8)}


def plot_bar(data, file_name):
    metrics = ['protein_protein', 'molecule_molecule', 'protein_both', 'molecule_both']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=file_name_to_figsize[file_name])
    ax = axes.ravel()
    for i, metric in enumerate(metrics):
        labels = data["name"]
        means = data[f"{metric}_mean"] * 100
        stds = data[f"{metric}_std"] * 100
        max_std = stds.max()
        x_min = means.min() - 3 * max_std
        x_max = means.max() + 3 * max_std
        bar_positions = range(len(labels))

        bars = ax[i].barh(bar_positions, means, xerr=stds, capsize=3, alpha=0.7)

        ax[i].set_title(metric_to_name[metric])
        ax[i].set_yticks(bar_positions)
        ax[i].set_yticklabels(labels, ha='right')
        ax[i].set_xlim(x_min, x_max)  # Set x-axis limit
        for bar, std in zip(bars, stds):
            bar_width = bar.get_width()
            annotation_x = bar_width + std  # Position annotation at std distance + 5 from bar end
            ax[i].annotate(f"{int(bar_width)}", xy=(annotation_x, bar.get_y() + bar.get_height() / 2),
                           xytext=(0, 0), textcoords="offset points", ha='left', va='center')
    fig.suptitle("ROC AUC Score", size=file_name_to_figsize[file_name][0] + 4)
    fig.tight_layout()

    plt.savefig(f"{file_name}.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    for file_name in ["fuse_configs.csv", "fuse_methods.csv", "train_vs_freeze.csv"]:
        data = pd.read_csv(file_name, sep="\t")
        with plt.style.context('tableau-colorblind10', after_reset=True):
            plot_bar(data, file_name.replace(".csv", ""))
