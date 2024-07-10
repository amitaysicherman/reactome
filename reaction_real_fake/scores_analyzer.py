import pandas as pd
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--print_count", type=int, default=0)
args = parser.parse_args()

our_key = 2
pre_key = 1
df = pd.read_csv(f"data/scores/rrf.csv")
metric = "auc"


if args.print_count:
    print(
        pd.pivot_table(df, index=['protein_emd', 'mol_emd'], columns=['pretrained_method'], values=metric, aggfunc="count"))

p_mean = pd.pivot_table(df, index=['protein_emd', 'mol_emd'], columns=['pretrained_method'], values=metric,
                        aggfunc="mean")
p_std = pd.pivot_table(df, index=['protein_emd', 'mol_emd'], columns=['pretrained_method'], values=metric,
                       aggfunc="std")
our_mean, our_std = p_mean[our_key], p_std[our_key]
pre_mean, pre_std = p_mean[pre_key], p_std[pre_key]
res = (p_mean * 100).round(1).astype(str) + "(" + (p_std * 100).round(2).astype(str) + ")"


def calcualte_ttest(data):
    our = data[data['conf'] == our_key][metric].values
    pre = data[data['conf'] == pre_key][metric].values
    return scipy.stats.ttest_ind(our, pre).pvalue


p_values = df.groupby(['protein_emd', 'mol_emd']).apply(calcualte_ttest)
res = res[[pre_key, our_key]]

res["statistically significant"] = p_values < 0.05
print(metric)
res = res.rename(columns={our_key: "Our", pre_key: "Pre-trained"})
output_file = f"data/scores/rrf_{metric}.csv"
res = res.reset_index()
res.to_csv("output_file")
print(res)
