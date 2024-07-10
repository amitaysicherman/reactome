import pandas as pd
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_model", type=int, default=0)
parser.add_argument("--task", type=str, default="BACE")
parser.add_argument("--print_count", type=int, default=0)
args = parser.parse_args()

our_key = 'True | True' if args.use_model else 'True | False'
pre_key = 'False | True'
df = pd.read_csv(f"data/scores/mol_{args.task}.csv")
df['molecule_model'] = df.name.apply(lambda x: x.split("-")[1])
df['conf'] = df['use_fuse'].astype(str) + " | " + df['use_model'].astype(str)

metric = "acc"#real is auc it;s bug
p_mean = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                        aggfunc="mean")
p_std = pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                       aggfunc="std")

if args.print_count:
    print(pd.pivot_table(df, index=['molecule_model'], columns=['conf'], values=metric,
                   aggfunc="count"))
our_mean, our_std = p_mean[our_key], p_std[our_key]
pre_mean, pre_std = p_mean[pre_key], p_std[pre_key]
res = (p_mean * 100).round(1).astype(str) + "(" + (p_std * 100).round(2).astype(str) + ")"


def calcualte_ttest(data):
    our = data[data['conf'] == our_key][metric].values
    pre = data[data['conf'] == pre_key][metric].values
    return scipy.stats.ttest_ind(our, pre).pvalue


p_values = df.groupby(['molecule_model']).apply(calcualte_ttest)
res = res[[pre_key, our_key]]

res["statistically significant"] = p_values < 0.05
print(metric)
res = res.rename(columns={our_key: "Our", pre_key: "Pre-trained"})
output_file = f"data/scores/go-{args.task}_{metric}.csv"
res = res.reset_index()
res.to_csv(output_file)
print(res.to_csv())
print(res)
