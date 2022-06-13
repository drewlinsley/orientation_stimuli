import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api as sm


if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    raise RuntimeError("You need to supply a kapadia-95 filename.")

# Config
file_dir = "model_outputs"
stim_orientation = 0
paper_dir = "digitized_data/drew"
paper_path_0 = os.path.join(paper_dir, "kapadia_fig11_s0.npy")
paper_path_1 = os.path.join(paper_dir, "kapadia_fig11_s1.npy")

# Create folder
output_dir = "results_kapadia_fig11"
os.makedirs(output_dir, exist_ok=True)

# Prepare figure
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f, axs = plt.subplots(1, 3, figsize=(8, 4))
plt.subplots_adjust(left=0.1,
                    bottom=0.3,
                    right=0.9,
                    top=0.9,
                    wspace=0.6,
                    hspace=0.4)
plt.suptitle("Kapadia '95 Figure 11")
# f.text(0.5, 0.01, "Populations' preferred orientation", ha='center')

# Load model data
model_data = np.load(os.path.join(file_dir, file_name))[stim_orientation]

# Load paper data
paper_data_0 = np.load(paper_path_0)
paper_data_1 = np.load(paper_path_1)

# Plot the results
axs[0].set_ylabel("Spikes/sec")
axs[0].bar(np.arange(len(paper_data_0)), paper_data_0, width=0.5, color="#d1040e")
axs[0].get_xaxis().set_ticklabels([])

axs[1].set_ylabel("Spikes/sec")
axs[1].bar(np.arange(len(paper_data_1)), paper_data_1, width=0.5, color="#d1040e")
axs[1].get_xaxis().set_ticklabels([])

axs[2].set_ylabel("Normalized firing rate")
axs[2].bar(np.arange(len(model_data)), model_data, width=0.5, color="#0092d6")
axs[2].get_xaxis().set_ticklabels([])

plt.savefig(os.path.join(output_dir, "{}_model.pdf".format(file_name)))
plt.show()

# Measure the similarity
bias = np.ones_like(paper_data_0.reshape(-1, 1))
X = np.concatenate((
    bias,
    model_data.reshape(-1, 1),
), -1)
clf = sm.OLS(paper_data_0.reshape(-1, 1), X).fit()
paper_sim_0 = clf.rsquared

bias = np.ones_like(paper_data_1.reshape(-1, 1))
X = np.concatenate((
    bias,
    model_data.reshape(-1, 1),
), -1)
clf = sm.OLS(paper_data_1.reshape(-1, 1), X).fit()
paper_sim_1 = clf.rsquared

mean_sim = (paper_sim_0 + paper_sim_1) / 2
np.save(
    os.path.join(output_dir, "{}_diff_scores".format(file_name)), [mean_sim, file_name])
print("Kapadia {} r^2: {}".format(file_name, mean_sim))

