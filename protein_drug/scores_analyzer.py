import pandas as pd
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_model", type=int, default=0)
parser.add_argument("--dataset", type=str, default="DrugBank")
parser.add_argument("--metric", type=str, default="aupr")
parser.add_argument("--print_count", type=int, default=0)
parser.add_argument("--mul_fuse", type=int, default=1)

args = parser.parse_args()

our_key = 'True | True | True | True' if args.use_model else 'True | True | False | False'
pre_key = 'False | False | True | True'
df = pd.read_csv(f"data/scores/drug_protein_{args.dataset}.csv")

if args.mul_fuse:
    df['name_len']=df.name.apply(lambda x: len(x.split("_")))
    df = df[df.name_len == 4]
    df['fuse_model'] = df.name.apply(lambda x: x.split("_")[1])
    df['protein_model'] = df.name.apply(lambda x: x.split("_")[2])
    df['molecule_model'] = df.name.apply(lambda x: x.split("_")[3])
    group_cols = ['fuse_model', 'protein_model', 'molecule_model']
else:
    df['protein_model'] = df.name.apply(lambda x: x.split("_")[1])
    df['molecule_model'] = df.name.apply(lambda x: x.split("_")[2])
    group_cols = ['protein_model', 'molecule_model']
df['conf'] = df['m_fuse'].astype(str) + " | " + df['p_fuse'].astype(str) + " | " + df['m_model'].astype(str) + " | " + \
             df['p_model'].astype(str)

metric = args.metric

if args.print_count:
    print(
        pd.pivot_table(df, index=group_cols, columns=['conf'], values=metric, aggfunc="count"))

p_mean = pd.pivot_table(df, index=group_cols, columns=['conf'], values=metric,
                        aggfunc="mean")
p_std = pd.pivot_table(df, index=group_cols, columns=['conf'], values=metric,
                       aggfunc="std")
our_mean, our_std = p_mean[our_key], p_std[our_key]
pre_mean, pre_std = p_mean[pre_key], p_std[pre_key]
res = (p_mean * 100).round(1).astype(str) + "(" + (p_std * 100).round(2).astype(str) + ")"


def calcualte_ttest(data):
    our = data[data['conf'] == our_key][metric].values
    pre = data[data['conf'] == pre_key][metric].values
    return scipy.stats.ttest_ind(our, pre).pvalue


p_values = df.groupby(group_cols).apply(calcualte_ttest)
res = res[[pre_key, our_key]]

res["statistically significant"] = p_values < 0.05
print(metric)
res = res.rename(columns={our_key: "Our", pre_key: "Pre-trained"})
output_file = f"data/scores/drug_protein_{args.dataset}_{metric}.csv"
res = res.reset_index()
print(res.to_csv())
res.to_csv("output_file")
print(res)
