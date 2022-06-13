import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api as sm


if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    raise RuntimeError("You need to supply a Kinoshita '09 filename.")

# Config
file_dir = "model_outputs"
stim_orientation = 0
paper_dir = "digitized_data/drew"
paper_path_0 = os.path.join(paper_dir, "kinoshita_fig11_s0.npy")

# Create folder
output_dir = "results_kinoshita_fig11"
os.makedirs(output_dir, exist_ok=True)

# Prepare figure
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.subplots_adjust(left=0.1,
                    bottom=0.3,
                    right=0.9,
                    top=0.9,
                    wspace=0.6,
                    hspace=0.4)
plt.suptitle("Kinoshita '09 Figure 11")
# f.text(0.5, 0.01, "Populations' preferred orientation", ha='center')

# Load model data
model_data = np.load(os.path.join(file_dir, file_name))[stim_orientation]
plt.subplot(121);plt.plot(np.load(paper_path_0));plt.subplot(122);plt.plot(model_data);plt.show()
model_data = model_data[:6]

# Normalize model data
model_data = model_data - 0.02
model_data = model_data[1:] / model_data[0]

# Load paper data
paper_data_0 = np.load(paper_path_0)

# # Normalize paper data
# paper_data_0 = paper_data_0 - 5.0
# paper_data_0 = paper_data_0[1:] / paper_data_0[0]

# Plot the results
axs[0].set_ylabel("Spikes/sec")
# axs[0].bar(np.arange(len(paper_data_0)), paper_data_0, width=0.5, color="#d1040e")
axs[0].plot(np.arange(len(paper_data_0)), paper_data_0, marker="o", color="#d1040e")
axs[0].axhline(y=1., color="black", linestyle="--", alpha=0.7)
axs[0].set_ylim([0.6, 1.5])

axs[1].set_ylabel("Normalized firing rate")
# axs[1].bar(np.arange(len(model_data)), model_data, width=0.5, color="#0092d6")
axs[1].plot(np.arange(len(model_data)), model_data, marker="o", color="#0092d6")
axs[1].axhline(y=1., color="black", linestyle="--", alpha=0.7)
axs[1].set_ylim([0.6, 1.5])

plt.savefig(os.path.join(output_dir, "{}_model.pdf".format(file_name)))
plt.show()

# Measure the similarity
bias = np.ones_like(paper_data_0.reshape(-1, 1))
X = np.concatenate((
    bias,
    model_data.reshape(-1, 1),
), -1)
clf = sm.OLS(paper_data_0.reshape(-1, 1), X).fit()
mean_sim = clf.rsquared
np.save(
    os.path.join(output_dir, "{}_diff_scores".format(file_name)), [mean_sim, file_name])
print("Kinoshita {} r^2: {}".format(file_name, mean_sim))

