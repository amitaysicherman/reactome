import pandas as pd
import scipy
import argparse
from common.data_types import NAME_TO_UI, MOL_UI_ORDER

tasks = ["CC", "BP", "MF"]

OUR = "Our"
PRE = "Pre-trained"
STAT = "statistically significant"


def calcualte_ttest(data, our_key, pre_key, metric):
    our = data[data['conf'] == our_key][metric].values
    pre = data[data['conf'] == pre_key][metric].values
    return scipy.stats.ttest_ind(our, pre).pvalue


def main(args):
    our_key = 'True | True' if args.use_model else 'True | False'
    pre_key = 'False | True'
    df = pd.read_csv(f"data/scores/go-{args.task}.csv")
    df['protein_model'] = df.name.apply(lambda x: x.split("_")[1] if len(x.split("_")) == 3 else "")
    df['conf'] = df['use_fuse'].astype(str) + " | " + df['use_model'].astype(str)

    metric = "f1max"
    p_mean = pd.pivot_table(df, index=['protein_model'], columns=['conf'], values=metric,
                            aggfunc="mean")
    p_std = pd.pivot_table(df, index=['protein_model'], columns=['conf'], values=metric,
                           aggfunc="std")

    if args.print_count:
        print(pd.pivot_table(df, index=['protein_model'], columns=['conf'], values=metric,
                             aggfunc="count"))
    res = (p_mean * 100).round(2).astype(str) + "\pm " + (p_std * 100).round(2).astype(str)
    p_values = df.groupby(['protein_model']).apply(calcualte_ttest, our_key=our_key, pre_key=pre_key, metric=metric)
    res = res[[pre_key, our_key]]

    res[f'{STAT}_{task}'] = p_values < 0.05

    res = res.rename(columns={our_key: f'{OUR}_{task}', pre_key: f'{PRE}_{task}'})

    res = res.reset_index()
    return res


def to_latex(res):
    for i, _ in res.iterrows():
        for task in tasks:
            if res.loc[i, f"{STAT}_{task}"]:
                a = float(res.loc[i, f"{OUR}_{task}"].split("\pm")[0])
                b = float(res.loc[i, f"{PRE}_{task}"].split("\pm")[0])
                if a > b:
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
    print(res.to_latex(index_names=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", type=int, default=0)
    parser.add_argument("--task", type=str, default="BP")
    parser.add_argument("--print_count", type=int, default=0)
    args = parser.parse_args()

    all_res = {}
    for task in tasks:
        args.task = task
        res = main(args)
        res.set_index('molecule_model', inplace=True, drop=True)
        all_res[task] = res
    final_res = all_res[tasks[0]]
    for task in tasks[1:]:
        final_res = final_res.merge(all_res[task], how='outer', left_index=True, right_index=True)
    print(final_res.to_csv())
    to_latex(final_res)
    print(final_res)
