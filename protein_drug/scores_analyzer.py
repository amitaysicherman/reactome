import pandas as pd
import scipy
import argparse
from common.path_manager import data_path
from common.data_types import NAME_TO_UI, MOL_UI_ORDER, PROT_UI_ORDER

datasets = ["DrugBank", "Davis", "KIBA"]
OUR = "Our"
PRE = "Pre-trained"
STAT = "statistically significant"


def main(args, ds):
    our_key = 'True | True | True | True' if args.use_model else 'True | True | False | False'
    pre_key = 'False | False | True | True'

    df = pd.read_csv(f"{data_path}/scores/drug_protein_{ds}.csv")
    df['name_len'] = df.name.apply(lambda x: len(x.split("_")))
    if args.mul_fuse:
        df = df[df.name_len == 4]
        df['fuse_model'] = df.name.apply(lambda x: x.split("_")[1])
        df['protein_model'] = df.name.apply(lambda x: x.split("_")[2])
        df['molecule_model'] = df.name.apply(lambda x: x.split("_")[3])
        group_cols = ['fuse_model', 'protein_model', 'molecule_model']
    else:
        df = df[df.name_len == 3]
        df['protein_model'] = df.name.apply(lambda x: x.split("_")[1])
        df['molecule_model'] = df.name.apply(lambda x: x.split("_")[2])
        group_cols = ['protein_model', 'molecule_model']
    df['conf'] = df['m_fuse'].astype(str) + " | " + df['p_fuse'].astype(str) + " | " + df['m_model'].astype(
        str) + " | " + \
                 df['p_model'].astype(str)

    metric = args.metric

    if args.print_count:
        print(
            pd.pivot_table(df, index=group_cols, columns=['conf'], values=metric, aggfunc="count"))

    p_mean = pd.pivot_table(df, index=group_cols, columns=['conf'], values=metric,
                            aggfunc="mean")
    p_std = pd.pivot_table(df, index=group_cols, columns=['conf'], values=metric,
                           aggfunc="std")
    res = (p_mean * 100).round(2).astype(str) + "(" + (p_std * 100).round(2).astype(str) + ")"

    def calcualte_ttest(data):
        our = data[data['conf'] == our_key][metric].values
        pre = data[data['conf'] == pre_key][metric].values
        return scipy.stats.ttest_ind(our, pre).pvalue

    p_values = df.groupby(group_cols).apply(calcualte_ttest)
    res = res[[pre_key, our_key]]

    res[f'{STAT}_{ds}'] = p_values < 0.05
    print(metric)
    res = res.rename(columns={our_key: f'{OUR}_{task}', pre_key: f'{PRE}_{task}'})
    res = res.reset_index()
    return res


def sort_by_order(x):
    prot, mol = x
    return PROT_UI_ORDER.index(prot), MOL_UI_ORDER.index(mol)


def to_latex(res):
    for i, _ in res.iterrows():
        for task in datasets:
            if res.loc[i, f"{STAT}_{task}"]:
                our = float(res.loc[i, f"{OUR}_{task}"].split("(")[0])
                pre = float(res.loc[i, f"{PRE}_{task}"].split("(")[0])
                if our > pre:
                    res.loc[i, f"{OUR}_{task}"] = "\\textbf{" + res.loc[i, f"{OUR}_{task}"] + "}"
                else:
                    res.loc[i, f"{PRE}_{task}"] = "\\textbf{" + res.loc[i, f"{PRE}_{task}"] + "}"
                print(our, pre)
    for task in datasets:
        res.drop(columns=f"{STAT}_{task}", inplace=True)

    res.columns = pd.MultiIndex.from_tuples(
        [(x.split("_")[1], x.split("_")[0]) for x in res.columns],
        names=['Task', 'Method'])
    res = res.T
    res = res[sorted(res.columns, key=sort_by_order)]
    res.columns = [f'{NAME_TO_UI[x[0]]} & {NAME_TO_UI[x[1]]}' for x in res.columns]
    print(res.T.reset_index().to_latex(index=False).replace("molecule_model", "").replace("protein_model", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="DrugBank")
    parser.add_argument("--metric", type=str, default="auprc")
    parser.add_argument("--print_count", type=int, default=0)
    parser.add_argument("--mul_fuse", type=int, default=0)
    args = parser.parse_args()
    all_res = dict()

    for task in datasets:
        res = main(args, task)
        res.set_index(['protein_model', 'molecule_model'], inplace=True, drop=True)
        all_res[task] = res
    final_res = all_res[datasets[0]]
    for task in datasets[1:]:
        final_res = final_res.merge(all_res[task], how='outer', left_index=True, right_index=True)
    print(final_res.to_csv())
    to_latex(final_res)
    print(final_res)
