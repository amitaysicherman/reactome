from common.utils import reaction_from_str
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from common.path_manager import reactions_file,figures_path
sns.set()
colors = sns.color_palette("tab10")
blue = colors[0]
green = colors[2]
red = colors[3]

with open(reactions_file) as f:
    lines = f.readlines()

years = [reaction_from_str(line).date.year for line in lines]
cutoff_date = np.quantile([reaction_from_str(line).date for line in lines], 0.8)
print(cutoff_date)
split_quantile = np.quantile(years, 0.8)

unknown_years = [year for year in years if year == 1970]
train_years = [year for year in years if year <= split_quantile and year != 1970]
test_years = [year for year in years if year > split_quantile]

unknown_years_unique, unknown_counts = np.unique(unknown_years, return_counts=True)
train_years_unique, train_counts = np.unique(train_years, return_counts=True)
test_years_unique, test_counts = np.unique(test_years, return_counts=True)

plt.figure(figsize=(10, 4))

plt.bar(train_years_unique, train_counts, label='Training', color=blue)
plt.bar(test_years_unique, test_counts, label='Testing', color=green)
plt.bar(2002, unknown_counts, label='Unknown Date', color=red)
props = dict(boxstyle='round', alpha=0.5, facecolor='white')

plt.text(0.05, 0.95, f'Cutoff Date:\n{cutoff_date}', transform=plt.gca().transAxes,
        verticalalignment='top', bbox=props)

plt.xlabel('Years')
plt.ylabel('Reactions Counts')
plt.title('Training & Testing Data Split')
plt.legend()
plt.tight_layout()
output_file=os.path.join(figures_path,"train_test_dates.png")

plt.savefig(output_file, dpi=300)
plt.show()
