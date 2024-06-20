from common.utils import reaction_from_str
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataset.dataset_builder import get_reactions
from common.path_manager import figures_path


def get_years(reactions):
    unknown_count = len([reaction.date.year for reaction in reactions if reaction.date.year == 1970])
    reactions = [reaction for reaction in reactions if reaction.date.year != 1970]
    years = [reaction.date.year for reaction in reactions]
    return years, unknown_count


sns.set()
colors = sns.color_palette("tab10")
blue = colors[0]
orange = colors[1]
green = colors[2]
red = colors[3]

train_lines, valid_lines, test_lines = get_reactions()
train_years, unknown_train = get_years(train_lines)
valid_years, unknown_valid = get_years(valid_lines)
test_years, unknown_test = get_years(test_lines)
unknown_counts = sum([unknown_train, unknown_valid, unknown_test])
train_valid_cutoff = valid_lines[0].date
valid_test_cutoff = test_lines[0].date

train_years_unique, train_counts = np.unique(train_years, return_counts=True)
valid_years_unique, valid_counts = np.unique(valid_years, return_counts=True)
test_years_unique, test_counts = np.unique(test_years, return_counts=True)

plt.figure(figsize=(10, 4))

plt.bar(train_years_unique, train_counts, label='Training', color=blue)
plt.bar(valid_years_unique, valid_counts, label='Validation', color=orange)
plt.bar(test_years_unique, test_counts, label='Testing', color=green)
plt.bar(2002, unknown_counts, label='Unknown Date', color=red)
props = dict(boxstyle='round', alpha=0.5, facecolor='white')

text = f"Train/Validation Cutoff:{train_valid_cutoff}\nValidation/Test Cutoff:{valid_test_cutoff}"
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top', bbox=props)

plt.xlabel('Years')
plt.ylabel('Reactions Counts')
plt.title('Train')
plt.legend()
plt.tight_layout()
output_file = os.path.join(figures_path, "train_test_dates.png")

plt.savefig(output_file, dpi=300)
plt.show()
