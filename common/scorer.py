import numpy as np
from sklearn.metrics import roc_auc_score

from common.data_types import REAL


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
        self.ids = {class_name: [] for class_name in self.class_names}
        for class_name in self.class_names:
            setattr(self, f'{class_name}_auc', 0)

    def add(self, y, pred, out, loss, class_names=None, id_list=None):
        self.count += len(y)
        self.batch_count += 1
        self.loss += loss
        if class_names:
            for i in range(len(y)):
                class_name = class_names[i]
                id_ = id_list[i] if id_list is not None else -1
                self.true_probs[class_name].extend(y[i].tolist())
                self.pred_probs[class_name].extend(sigmoid(out[i]).tolist())
                self.ids[class_name].append(id_)
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

    def save_full_res(self, output_path_prefix):
        for class_name in self.class_names:
            with open(f"{output_path_prefix}_{self.name}_{class_name}.txt", "w") as f:
                for i in range(len(self.true_probs[class_name])):
                    f.write(
                        f"{self.ids[class_name][i]},{self.true_probs[class_name][i]:.3f},{self.pred_probs[class_name][i]:.3f}\n")

    def get_log(self):
        auc_dict = self.compute_auc_per_class()
        metrics = {
            f"{self.name}/loss": self.loss / self.batch_count,
            f'{self.name}/fake_all': auc_dict['all'],
        }
        for class_name in self.class_names:
            if class_name == REAL:
                continue
            name = f"{self.name}/{class_name}"
            metrics[f"{name}"] = auc_dict[class_name]
        self.reset()
        return metrics
