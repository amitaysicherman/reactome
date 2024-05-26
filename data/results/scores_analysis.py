import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pretrained_method_names = {
    0: "no",
    1: "pretrained",
    2: "fuse"
}
base_dir = "/data/results"
keys_to_show = ["train_all_emd", "pretrained_method"]


def name_to_ui(name, keys_to_show=keys_to_show):
    ui = ""
    name = name.replace(".txt", "")
    for key_value in name.split("-"):
        *key, value = key_value.split("_")
        key = "_".join(key)
        if keys_to_show and key not in keys_to_show:
            continue
        if key == "pretrained_method":
            value = pretrained_method_names[int(value)]
            ui += value
        if key == "train_all_emd":
            if value == "1":
                ui += " (trained)"
            else:
                ui += " (freezed)"
    return ui


def parse_file(file_name):
    with open(file_name) as f:
        lines = f.read().splitlines()
    data = [eval(line.replace('nan', "0")) for line in lines[1::2]]
    data = [{**data[i], **data[i + 1]} for i in range(0, len(data), 2)]
    data = pd.DataFrame(data)
    return data


def get_last_train_test(data):
    train = dict()
    test = dict()
    for col in data.columns:
        col_data = data[col].values
        value = np.min(col_data) if "loss" in col else np.max(col_data)
        type_, name = col.split("/")

        if type_ == "train":
            train[name] = value
        else:
            test[name] = value
    return train, test


metrics = ["loss", "all_auc", "fake_location_all", "fake_protein", "fake_molecule"]
train_data = {metric: dict() for metric in metrics}
test_data = {metric: dict() for metric in metrics}
epoch_data = {metric: dict(train=[], test=[]) for metric in metrics}

for file_name in os.listdir(base_dir):
    if not file_name.endswith(".txt"):
        continue
    data = parse_file(f"{base_dir}/{file_name}")
    train, test = get_last_train_test(data)
    ui = name_to_ui(file_name)
    for metric in metrics:
        train_data[metric][ui] = train[metric]
        test_data[metric][ui] = test[metric]
        epoch_data[metric]['train'].append(data.filter(like=f"train/{metric}"))
        epoch_data[metric]['test'].append(data.filter(like=f"test/{metric}"))


def plot_metric_bars(metric, train_data, test_data):
    with plt.style.context('tableau-colorblind10', after_reset=True):
        fig, (train_ax, test_ax) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(metric.capitalize(), fontsize=16)

        train_ax.set_title("Train", fontsize=14)
        test_ax.set_title("Test", fontsize=14)

        train_values = list(train_data[metric].values())
        train_keys = list(train_data[metric].keys())
        train_values = [x for _, x in sorted(zip(train_keys, train_values))]
        train_keys = sorted(train_keys)

        test_values = list(test_data[metric].values())
        test_keys = list(test_data[metric].keys())
        test_values = [x for _, x in sorted(zip(test_keys, test_values))]
        test_keys = sorted(test_keys)

        min_value = min(min(train_values), min(test_values)) - 0.03
        max_value = max(max(train_values), max(test_values)) + 0.05

        train_bars = train_ax.bar(train_keys, train_values, color='#1f77b4')
        test_bars = test_ax.bar(test_keys, test_values, color='#ff7f0e')

        train_ax.set_ylim(min_value, max_value)
        test_ax.set_ylim(min_value, max_value)

        train_ax.set_xticks(range(len(train_keys)))
        train_ax.set_xticklabels(train_keys, rotation=90, fontsize=10)

        test_ax.set_xticks(range(len(test_keys)))
        test_ax.set_xticklabels(test_keys, rotation=90, fontsize=10)

        # Annotate the bar values
        for bar in train_bars:
            yval = bar.get_height()

            train_ax.annotate(f'{yval:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, yval),
                              xytext=(0, 5),  # 5 points vertical offset
                              textcoords='offset points',
                              ha='center',
                              va='bottom',
                              fontsize=10,
                              color='black',
                              weight='bold')

        for bar in test_bars:
            yval = bar.get_height()
            test_ax.annotate(f'{yval:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, yval),
                             xytext=(0, 5),  # 5 points vertical offset
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             fontsize=10,
                             color='black',
                             # weight='bold'
                             )

        plt.tight_layout()
        plt.show()


def plot_metric_lines(metric, epoch_data):
    with plt.style.context('tableau-colorblind10', after_reset=True):
        fig, (train_ax, test_ax) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f"{metric.capitalize()} Over Epochs")

        train_ax.set_title("Train")
        test_ax.set_title("Test")

        for i, config in enumerate(epoch_data[metric]['train']):
            config_name = list(train_data[metric].keys())[i]
            train_ax.plot(config.index, config.values, label=config_name)
        for i, config in enumerate(epoch_data[metric]['test']):
            config_name = list(test_data[metric].keys())[i]
            test_ax.plot(config.index, config.values, label=config_name)

        train_ax.legend()
        test_ax.legend()

        plt.tight_layout()
        plt.show()


for metric in metrics:
    plot_metric_bars(metric, train_data, test_data)
    plot_metric_lines(metric, epoch_data)
