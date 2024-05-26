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
        self.reset()
        return metrics
