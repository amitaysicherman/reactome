import pandas as pd
import scipy
import argparse

tasks = ["BACE", "BBBP", "ClinTox", "HIV", "SIDER"]
OUR = "Our"
PRE = "Pre-trained"
STAT = "statistically significant"


def main(use_model, task, print_count) -> pd.DataFrame:
    our_key = 'True | True' if use_model else 'True | False'
    pre_key = 'False | True'
    df = pd.read_csv(f"data/scores/mol_{task}.csv")
    df['molecule_model'] = df.name.apply(lambda x: x.split("-")[1])
    df['conf'] = df['use_fuse'].astype(str) + " | " + df['use_model'].astype(str)

    metric = "acc"  # real is auc it;s bug
    if print_count:
        print(pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                             aggfunc="count"))

    res = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                         aggfunc="mean")

    def drop_dup_mis_seed(data):
        data = data.drop_duplicates(['seed', 'conf', 'molecule_model'])
        seeds = []
        for s in data['seed'].unique():
            d = data[data['seed'] == s]
            d_our = d[d['conf'] == our_key]
            d_pre = d[d['conf'] == pre_key]
            if len(d_our) and len(d_pre):
                assert len(d_our) == len(d_pre) == 1
                seeds.append(s)
            else:
                print(f"seed {s} is missing {task}")
        data = data[data['seed'].isin(seeds)]
        return data.sort_values(by="seed")

    def def_delta_std(data):
        data = drop_dup_mis_seed(data)
        our = data[data['conf'] == our_key][metric].values
        pre = data[data['conf'] == pre_key][metric].values
        delta = (our - pre)
        return delta.std()

    res[f"Delta STD_{task}"] = df.groupby(['molecule_model']).apply(def_delta_std)
    res = (res * 100).round(2).astype(str)

    def calcualte_ttest(data):
        data = drop_dup_mis_seed(data)
        our = data[data['conf'] == our_key][metric].values
        pre = data[data['conf'] == pre_key][metric].values
        return scipy.stats.ttest_rel(our, pre).pvalue

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
                if res.loc[i, f"{OUR}_{task}"] > res.loc[i, f"{PRE}_{task}"]:
                    res.loc[i, f"{OUR}_{task}"] = "\\textbf{" + res.loc[i, f"{OUR}_{task}"] + "}"
                else:
                    res.loc[i, f"{PRE}_{task}"] = "\\textbf{" + res.loc[i, f"{PRE}_{task}"] + "}"
    for task in tasks:
        res.drop(columns=f"{STAT}_{task}", inplace=True)
    res.columns = pd.MultiIndex.from_tuples([(x.split("_")[1], x.split("_")[0]) for x in res.columns],
                                            names=['Task', 'Method'])

    print(res.T.to_latex())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_model", type=int, default=0)
    parser.add_argument("--task", type=str, default="BACE")
    parser.add_argument("--print_count", type=int, default=0)
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
