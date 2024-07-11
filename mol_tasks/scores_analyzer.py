import pandas as pd
import scipy
import argparse


def main(use_model, task, print_count) -> pd.DataFrame:
    our_key = 'True | True' if use_model else 'True | False'
    pre_key = 'False | True'
    df = pd.read_csv(f"data/scores/mol_{task}.csv")
    df['molecule_model'] = df.name.apply(lambda x: x.split("-")[1])
    df['conf'] = df['use_fuse'].astype(str) + " | " + df['use_model'].astype(str)

    metric = "acc"  # real is auc it;s bug
    p_mean = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                            aggfunc="mean")
    p_std = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                           aggfunc="std")

    if print_count:
        print(pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                             aggfunc="count"))
    res = (p_mean * 100).round(2).astype(str) + "(" + (p_std * 100).round(2).astype(str) + ")"

    def calcualte_ttest(data):
        our = data[data['conf'] == our_key][metric].values
        pre = data[data['conf'] == pre_key][metric].values
        return scipy.stats.ttest_ind(our, pre).pvalue

    p_values = df.groupby(['molecule_model']).apply(calcualte_ttest)
    res = res[[pre_key, our_key]]

    res["statistically significant"] = p_values < 0.05
    print(metric)
    res = res.rename(columns={our_key: "Our", pre_key: "Pre-trained"})
    res = res.reset_index()
    print(res.to_csv())
    print(res)
    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", type=int, default=0)
    parser.add_argument("--task", type=str, default="BACE")
    parser.add_argument("--print_count", type=int, default=0)
    args = parser.parse_args()
    all_res = dict()
    tasks = ["BACE", "BBBP", "ClinTox", "HIV", "SIDER"]
    for task in tasks:
        res = main(args.use_model, task, args.print_count)
        res.set_index('molecule_model', inplace=True, drop=True)
        all_res[task] = res
    final_res = all_res[tasks[0]]
    for task in tasks[1:]:
        final_res = final_res.merge(all_res[task], how='outer', left_index=True, right_index=True, suffixes=('', f'_{task}'))
    print(final_res.to_latex())
    print(final_res.to_csv())
