import pandas as pd
from scipy.stats import ttest_ind
from common.data_types import NAME_TO_UI, MOL_UI_ORDER, PROT_UI_ORDER
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=str, default="NO")
args = parser.parse_args()

# Configuration Constants
our = "our"
both = "both"
pre = "pre"
SELECTED_METRIC = "selected_metric"

# Define columns and mappings
index_cols = ['task_name', 'protein_emd', 'mol_emd']
conf = ['conf']
metrics_cols = ['mse', 'mae', 'r2', 'pearsonr', 'spearmanr', 'auc', 'auprc', 'acc', 'f1_max']
all_cols = index_cols + conf + metrics_cols
conf_cols = ['pre', 'our', 'both']
type_to_metric = {'M': 'acc', 'P': "pearsonr", 'PD': "auprc", 'PDA': "pearsonr", 'PPI': "auprc", 'PPIA': "pearsonr"}

# Mapping from task names to types
name_to_type_dict = {
    'BetaLactamase': "P", 'Fluorescence': "P", 'Stability': "P", 'Solubility': "P",
    'HumanPPI': "PPI", 'YeastPPI': "PPI", 'PPIAffinity': "PPIA", 'BindingDB': "PDA",
    'PDBBind': "PDA", 'BACE': "M", 'BBBP': "M", 'ClinTox': "M", 'HIV': "M",
    'SIDER': "M", 'Tox21': "M", 'DrugBank': "PD", 'Davis': "PD", 'KIBA': "PD"
}

TYPE_TO_NAME = {
    'P': 'Protein Function Prediction', 'PPI': 'Protein-Protein Interaction Prediction',
    "PPIA": 'Protein-Protein Interaction Affinity Prediction',
    'M': 'Molecule Property Prediction', 'PD': 'Protein-Drug Interaction Prediction',
    "PDA": 'Protein-Drug Interaction Affinity Prediction'
}
METRIC_TO_NAME = {
    'mse': 'Mean Squared Error', 'mae': 'Mean Absolute Error', 'r2': 'R2', 'pearsonr': 'Pearson Correlation',
    'spearmanr': 'Spearman Correlation', 'auc': 'Area Under the ROC Curve (AUC)',
    'auprc': 'Area Under the PR Curve (AUPRC)',
    'acc': 'Accuracy', 'f1_max': 'F1 Max Score'
}

COLS_TO_NAME = {
    'task_name': 'Task', 'protein_emd': 'Protein Embedding', 'mol_emd': 'Molecule Embedding'
}


def name_to_type(x):
    return name_to_type_dict[x.split("_")[0]]


def task_to_selected_matic(task):
    if task in name_to_type_dict:
        return type_to_metric[name_to_type_dict[task]]
    elif task.split("_")[0] in name_to_type_dict:
        return type_to_metric[name_to_type(task.split("_")[0])]
    else:
        return None


def df_to_selected_matic(df):
    bool_filter = []
    for i, row in df.iterrows():
        metric = task_to_selected_matic(row['task_name'])
        if metric is None:
            bool_filter.append(False)
            continue
        else:
            bool_filter.append(True)
        if metric in ['mse', 'mae']:
            df.loc[i, SELECTED_METRIC] = -1 * df.loc[i, metric]
        else:
            df.loc[i, SELECTED_METRIC] = df.loc[i, metric]
    df = df[bool_filter]
    return df


def round_num(x):
    if pd.isna(x):
        return 0

    return abs(round(x * 100, 2))  # for mse and mae


def get_format_results_agg(group):
    # Extract values for each configuration
    pre_values = group[group['conf'] == pre][SELECTED_METRIC]
    ablation_data = group[group["ablation"] == args.ablation]
    our_values = ablation_data[ablation_data['conf'] == our][SELECTED_METRIC]
    both_values = ablation_data[ablation_data['conf'] == both][SELECTED_METRIC]
    # Calculate mean and standard deviation for each configuration
    pre_mean, pre_std = pre_values.mean(), pre_values.std()
    our_mean, our_std = our_values.mean(), our_values.std()
    both_mean, both_std = both_values.mean(), both_values.std()

    # Determine whether lower values are better for the metric
    lower_is_better_metrics = {'mse', 'mae'}

    # Determine the best configuration based on the metric
    if SELECTED_METRIC in lower_is_better_metrics:
        best_mean, best_std = (our_mean, our_std) if our_mean < both_mean else (both_mean, both_std)
        best_values = our_values if our_mean < both_mean else both_values
    else:
        best_mean, best_std = (our_mean, our_std) if our_mean > both_mean else (both_mean, both_std)
        best_values = our_values if our_mean > both_mean else both_values

    # Perform t-test between 'pre' and 'best'
    t_stat, p_value = ttest_ind(pre_values, best_values, equal_var=False, nan_policy='omit')

    # Check if the difference is statistically significant
    significant = p_value < 0.05

    # Format the results with potential bolding for statistical significance
    if significant:
        if (SELECTED_METRIC in lower_is_better_metrics and pre_mean > best_mean) or \
                (SELECTED_METRIC not in lower_is_better_metrics and pre_mean < best_mean):
            # If the best is significantly better, bold the best
            best_result = f"\\textbf{{{round_num(best_mean)}}}({round_num(best_std)})"
            pre_result = f"{round_num(pre_mean)}({round_num(pre_std)})"
        else:
            # If pre is significantly better, bold pre
            best_result = f"{round_num(best_mean)}({round_num(best_std)})"
            pre_result = f"\\textbf{{{round_num(pre_mean)}}}({round_num(pre_std)})"
    else:
        # No significant difference, just format normally
        pre_result = f"{round_num(pre_mean)}({round_num(pre_std)})"
        best_result = f"{round_num(best_mean)}({round_num(best_std)})"

    return pre_result, best_result


def add_ablation_col(data):
    data['ablation'] = data['task_name'].apply(lambda x: x.split("_")[1] if len(x.split("_")) > 1 else 'NO')
    data['task_name'] = data['task_name'].apply(lambda x: x.split("_")[0])
    return data


# Load data
data = pd.read_csv("data/scores/torchdrug.csv", on_bad_lines='warn')
data = data.dropna()
data = add_ablation_col(data)
data = df_to_selected_matic(data)

# Group by and apply aggregation
format_results = data.groupby(index_cols).apply(get_format_results_agg)

# Convert the results to a DataFrame for easy handling
format_results_df = pd.DataFrame(format_results.tolist(), columns=['Pretrained Models', 'Our'],
                                 index=format_results.index)

# Display the first 20 rows of the results
format_results_df = format_results_df.reset_index()
format_results_df['task_type'] = format_results_df['task_name'].apply(name_to_type)
format_results_df['protein_emd'] = format_results_df['protein_emd'].apply(lambda x: NAME_TO_UI[x])
format_results_df['mol_emd'] = format_results_df['mol_emd'].apply(lambda x: NAME_TO_UI[x])
format_results_df = format_results_df.sort_values(by=['task_type', 'task_name', 'protein_emd', 'mol_emd'])


def print_format_latex(data: pd.DataFrame):
    tast_type = data['task_type'].iloc[0]
    caption = f'{METRIC_TO_NAME[type_to_metric[tast_type]]},{TYPE_TO_NAME[tast_type]}'
    label = f'tab:{tast_type}_results'
    index_cols_print = index_cols[:]
    data = data.drop(columns=['task_type'])
    if data['task_name'].nunique() == 1:
        data = data.drop(columns=['task_name'])
        index_cols_print.remove('task_name')
    if data['protein_emd'].nunique() == 1:
        data = data.drop(columns=['protein_emd'])
        index_cols_print.remove('protein_emd')
    if data['mol_emd'].nunique() == 1:
        data = data.drop(columns=['mol_emd'])
        index_cols_print.remove('mol_emd')
    data.rename(columns=COLS_TO_NAME, inplace=True)
    index_cols_print = [COLS_TO_NAME[x] for x in index_cols_print]
    data = data.set_index(index_cols_print)
    len_index = len(index_cols_print)
    col_format = 'l' * len_index + "|" + 'l' * len(data.columns)
    print(data.to_latex(index=True, escape=False, caption=caption, label=label, column_format=col_format).replace(
        "begin{table}", "begin{table}\n\centering"))


format_results_df.groupby('task_type').apply(print_format_latex)
