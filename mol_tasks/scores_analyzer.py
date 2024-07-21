import pandas as pd
import scipy
import argparse
from common.data_types import NAME_TO_UI, MOL_UI_ORDER
from common.path_manager import data_path
tasks = ["BACE", "BBBP", "ClinTox", "HIV", "SIDER"]
OUR = "Our"
PRE = "Pre-trained"
STAT = "statistically significant"


def main(use_model, task, print_count) -> pd.DataFrame:
    our_key = 'True | True' if use_model else 'True | False'
    pre_key = 'False | True'
    df = pd.read_csv(f"{data_path}/scores/mol_{task}.csv")
    have_name = df.name.apply(lambda x: len(x.split("-")) > 1)
    print(len(df), have_name.sum())
    df = df[have_name]

    df['molecule_model'] = df.name.apply(lambda x: x.split("-")[1])

    df['conf'] = df['use_fuse'].astype(str) + " | " + df['use_model'].astype(str)

    metric = "acc"  # real is auc it;s bug
    if print_count:
        print(pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                             aggfunc="count"))


    p_mean = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                            aggfunc="mean")

    p_std = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                           aggfunc="std")

    if args.print_count:
        print(pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                             aggfunc="count"))


    res = (p_mean * 100).round(1).astype(str)+ "(" + (p_std * 100).round(1).astype(str) + ")"
    def calcualte_ttest(data):
        our = data[data['conf'] == our_key][metric].values
        pre = data[data['conf'] == pre_key][metric].values
        return scipy.stats.ttest_ind(our, pre).pvalue

    p_values = df.groupby(['molecule_model']).apply(calcualte_ttest)
    res = res[[pre_key, our_key]]

    res[f'{STAT}_{task}'] = p_values < 0.05

    res = res.rename(columns={our_key: f'{OUR}_{task}', pre_key: f'{PRE}_{task}'})

    res = res.reset_index()
    return res


def to_latex(res):
    for i, _ in res.iterrows():
        for task in tasks:
            if res.loc[i, f"{STAT}_{task}"]:
                our=float(res.loc[i, f"{OUR}_{task}"].split("(")[0])
                pre=float(res.loc[i, f"{PRE}_{task}"].split("(")[0])
                if our > pre:
                    res.loc[i, f"{OUR}_{task}"] = "\\textbf{" + res.loc[i, f"{OUR}_{task}"] + "}"
                else:
                    res.loc[i, f"{PRE}_{task}"] = "\\textbf{" + res.loc[i, f"{PRE}_{task}"] + "}"
    for task in tasks:
        res.drop(columns=f"{STAT}_{task}", inplace=True)

    res.columns = pd.MultiIndex.from_tuples(
        [(x.split("_")[1], x.split("_")[0]) for x in res.columns],
        names=['Task', 'Method'])
    res = res.T
    res = res[sorted(res.columns, key=MOL_UI_ORDER.index)]
    res.rename(columns={x: NAME_TO_UI[x] for x in res.columns}, inplace=True)
    print(res.T.to_latex().replace("molecule_model", ""))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", type=int, default=0)
    parser.add_argument("--task", type=str, default="BACE")
    parser.add_argument("--print_count", type=int, default=1)
    args = parser.parse_args()
    all_res = dict()
    for task in tasks:
        res = main(args.use_model, task, args.print_count)
        res.set_index('molecule_model', inplace=True, drop=True)
        all_res[task] = res
    final_res = all_res[tasks[0]]
    for task in tasks[1:]:
        final_res = final_res.merge(all_res[task], how='outer', left_index=True, right_index=True)
    print(final_res.to_csv())
    to_latex(final_res)
    print(final_res)
