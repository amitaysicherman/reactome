import pandas as pd
import scipy
import argparse
from common.data_types import NAME_TO_UI, PROT_UI_ORDER
from common.path_manager import data_path

datasets = ["loc0", "loc1"]
OUR = "Our"
PRE = "Pre-trained"
STAT = "statistically significant"


def main(args, ds) -> pd.DataFrame:
    our_key = 'True | True' if args.use_model else 'True | False'
    pre_key = 'False | True'
    df = pd.read_csv(f"{data_path}/scores/{args.task_output_prefix}{ds}.csv")
    have_name = df.name.apply(lambda x: len(x.split("_")) > 1)
    print(len(df), have_name.sum())
    df = df[have_name]

    df['protein_model'] = df.name.apply(lambda x: x.split("_")[1])

    df['conf'] = df['use_fuse'].astype(str) + " | " + df['use_model'].astype(str)

    metric = "acc"  # real is auc it;s bug

    p_mean = pd.pivot_table(df, index=['protein_model'], columns=['conf'], values=metric,
                            aggfunc="mean")

    p_std = pd.pivot_table(df, index=['protein_model'], columns=['conf'], values=metric,
                           aggfunc="std")

    if args.print_count:
        print(pd.pivot_table(df, index=['protein_model'], columns=['conf'], values=metric,
                             aggfunc="count"))


    res = (p_mean * 100).round(2).astype(str) + "(" + (p_std * 100).round(2).astype(str) + ")"

    def calcualte_ttest(data):
        our = data[data['conf'] == our_key][metric].values
        pre = data[data['conf'] == pre_key][metric].values
        return scipy.stats.ttest_ind(our, pre).pvalue

    p_values = df.groupby(['protein_model']).apply(calcualte_ttest)
    res = res[[pre_key, our_key]]

    res[f'{STAT}_{ds}'] = p_values < 0.05

    res = res.rename(columns={our_key: f'{OUR}_{ds}', pre_key: f'{PRE}_{ds}'})

    res = res.reset_index()
    return res


def to_latex(res):
    for i, _ in res.iterrows():
        for task in datasets:
            if res.loc[i, f"{STAT}_{task}"]:
                if res.loc[i, f"{OUR}_{task}"] > res.loc[i, f"{PRE}_{task}"]:
                    res.loc[i, f"{OUR}_{task}"] = "\\textbf{" + res.loc[i, f"{OUR}_{task}"] + "}"
                else:
                    res.loc[i, f"{PRE}_{task}"] = "\\textbf{" + res.loc[i, f"{PRE}_{task}"] + "}"
    for task in datasets:
        res.drop(columns=f"{STAT}_{task}", inplace=True)

    res.columns = pd.MultiIndex.from_tuples(
        [(x.split("_")[1], x.split("_")[0]) for x in res.columns],
        names=['Task', 'Method'])
    res = res.T
    res = res[sorted(res.columns, key=PROT_UI_ORDER.index)]
    res.rename(columns={x: NAME_TO_UI[x] for x in res.columns}, inplace=True)
    print(res.T.to_latex().replace("protein_model", ""))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", type=int, default=0)
    parser.add_argument("--task_output_prefix", type=str, default="")
    parser.add_argument("--print_count", type=int, default=0)
    args = parser.parse_args()
    all_res = dict()
    for ds in datasets:
        if ds == "loc1":
            args.use_model = 1
        res = main(args, ds)
        res.set_index('protein_model', inplace=True, drop=True)
        all_res[ds] = res
    final_res = all_res[datasets[0]]
    for task in datasets[1:]:
        final_res = final_res.merge(all_res[task], how='outer', left_index=True, right_index=True)
    print(final_res.to_csv())
    to_latex(final_res)
    print(final_res)

