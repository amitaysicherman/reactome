import pandas as pd
from common.data_types import NAME_TO_UI

data = pd.read_csv("data/scores/torchdrug.csv")
data = data.dropna()
index_cols = ['task_name', 'protein_emd', 'mol_emd']
conf = ['conf']
metrics_cols = ['mse', 'mae', 'r2', 'pearsonr', 'spearmanr', 'auc', 'auprc', 'acc', 'f1_max']
all_cols = index_cols + conf + metrics_cols
name_to_type = {'BetaLactamase': "P", 'Fluorescence': "P", 'Stability': "P", 'Solubility': "P", 'HumanPPI': "PPI",
                'YeastPPI': "PPI", 'PPIAffinity': "PPIA", 'BindingDB': "PDA", 'PDBBind': "PDA", 'BACE': "M",
                'BBBP': "M", 'ClinTox': "M", 'HIV': "M", 'SIDER': "M", 'Tox21': "M", 'DrugBank': "PD", 'Davis': "PD",
                'KIBA': "PD"}

type_to_metric = {'M': 'auprc', 'P': "mse", 'PD': "auprc", 'PDA': "pearsonr", 'PPI': "auprc", 'PPIA': "pearsonr"}

conf_cols = ['pre', 'our', 'both']
OUR_FINAL = "Our"
our = "our"
both = "both"
pre = "pre"
diff = "diff"

tasks_to_skip = ["YeastPPI"]
mean_values = data.groupby(index_cols + conf)[metrics_cols].mean().reset_index().pivot(index=index_cols, columns=conf,
                                                                                       values=metrics_cols).dropna()
std_values = data.groupby(index_cols + conf)[metrics_cols].std().reset_index().pivot(index=index_cols, columns=conf,
                                                                                     values=metrics_cols).dropna()

for m in metrics_cols:
    mean_values[[(m, x) for x in conf_cols]] = mean_values[[(m, x) for x in conf_cols]].round(4) * 100

    if m not in ['mse', 'mae']:
        diffa = mean_values[(m, our)] - mean_values[(m, pre)]
        diffb = mean_values[(m, both)] - mean_values[(m, pre)]
        mean_values[(m, OUR_FINAL)] = mean_values[[(m, both), (m, our)]].max(axis=1)
    else:
        diffa = mean_values[(m, pre)] - mean_values[(m, our)]
        diffb = mean_values[(m, pre)] - mean_values[(m, both)]
        mean_values[(m, OUR_FINAL)] = mean_values[[(m, both), (m, our)]].min(axis=1)
    mean_values[(m, diff)] = pd.concat([diffa, diffb], axis=1).max(axis=1)

for T in set(name_to_type.values()):

    task_data = mean_values.reset_index()

    task_data = task_data[task_data['task_name'].apply(lambda x: name_to_type[x] == T)].set_index(index_cols)
    print(T, len(task_data), type_to_metric[T])

    if len(task_data) == 0:
        continue
    for col in metrics_cols:
        print(col, sum(task_data[(col, diff)] > 0), task_data[(col, diff)].mean())
    print_data = task_data[
        [(type_to_metric[T], x) for x in conf_cols] + [(type_to_metric[T], diff)] + [(type_to_metric[T], OUR_FINAL)]]
    print_data.columns = [x[1] for x in print_data.columns]
    print(print_data)
    print_data[pre] = print_data[pre].round(2).astype(str)
    print_data[OUR_FINAL] = print_data[OUR_FINAL].round(2).astype(str)
    print_data = print_data.reset_index()
    print_data['protein_emd'] = print_data['protein_emd'].apply(lambda x: NAME_TO_UI[x])
    print_data['mol_emd'] = print_data['mol_emd'].apply(lambda x: NAME_TO_UI[x])
    index_to_use = index_cols[:]
    if print_data['protein_emd'].nunique() == 1:
        index_to_use.remove('protein_emd')
    if print_data['mol_emd'].nunique() == 1:
        index_to_use.remove('mol_emd')
    if print_data['task_name'].nunique() == 1:
        index_to_use.remove('task_name')

    print(print_data[index_to_use + [pre, OUR_FINAL]].to_latex(index=False, float_format=".2f").replace("protein_emd",
                                                                                                        "Protein").replace(
        "mol_emd", "Molecule").replace("task_name", "Task").replace("pre", "Pretrained"))

    print("--------")

    # print(mean_values)
