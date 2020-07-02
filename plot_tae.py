import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob


surround_file = glob(os.path.join("TAE_surround_outputs", "*.npz"))[0]
no_surround_file = glob(os.path.join("TAE_no_surround_outputs", "*.npz"))[0]
surround_60_file = glob(os.path.join("TAE_surround_60_outputs", "*.npz"))[0]
surround_90_file = glob(os.path.join("TAE_surround_90_outputs", "*.npz"))[0]

surround = np.load(surround_file, allow_pickle=True, encoding="latin1")  # noqa
surround_60 = np.load(surround_60_file, allow_pickle=True, encoding="latin1")  # noqa
surround_90 = np.load(surround_90_file, allow_pickle=True, encoding="latin1")  # noqa
no_surround = np.load(no_surround_file, allow_pickle=True, encoding="latin1")  # noqa
# surround_data = np.abs(surround['test_dict'][0]["logits"])
# no_surround_data = np.abs(no_surround['test_dict'][0]["logits"])
surround_data = np.abs(surround['test_dict'][0]["logits"])
surround_60_data = np.abs(surround_60['test_dict'][0]["logits"])
surround_90_data = np.abs(surround_90['test_dict'][0]["logits"])
no_surround_data = np.abs(no_surround['test_dict'][0]["logits"])

surround_deg = ((np.arctan2(surround_data[:, 0], surround_data[:, 1])) * 180 / np.pi) * -1  # noqa
surround_60_deg = ((np.arctan2(surround_60_data[:, 0], surround_60_data[:, 1])) * 180 / np.pi) * -1  # noqa
surround_90_deg = ((np.arctan2(surround_90_data[:, 0], surround_90_data[:, 1])) * 180 / np.pi) * -1  # noqa
no_surround_deg = ((np.arctan2(no_surround_data[:, 0], no_surround_data[:, 1])) * 180 / np.pi) * -1  # noqa

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

f, ax = plt.subplots(1, 1)
plt.plot(surround_data, label="TAE Experiment", color="black")
plt.plot(surround_60_data, label="Inducer only", color="#d40007", alpha=0.3)
plt.plot(surround_90_data, label="Probe only", color="blue", alpha=0.5)
plt.xlabel("Steps")
plt.ylabel("Decoded orientation")
# ax.axhline(y=-30, xmin=0, xmax=0.125, color="grey", linestyle="--")
# ax.axhline(y=0, xmin=0.125, xmax=1.0, color="grey", linestyle="--")
# plt.text(x=0.25, y=-33, s="60 degree adaptor")
# plt.text(x=7.5, y=1, s="90 degree probe")
# plt.plot(no_surround_deg, label="no surround")
plt.legend()
plt.show()
plt.close(f)


f, ax = plt.subplots(1, 1)
plt.plot(surround_deg, label="TAE Experiment", color="black")
plt.plot(surround_60_deg, label="Inducer only", color="#d40007", alpha=0.3)
# plt.plot(surround_90_deg, label="Probe only", color="blue", alpha=0.5)
plt.xlabel("Steps")
plt.ylabel("Decoded orientation")
ax.axhline(y=-30, xmin=0, xmax=0.125, color="grey", linestyle="--")
ax.axhline(y=0, xmin=0.125, xmax=1.0, color="grey", linestyle="--")
plt.text(x=0.25, y=-33, s="60 degree adaptor")
plt.text(x=7.5, y=1, s="90 degree probe")
# plt.plot(no_surround_deg, label="no surround")
plt.legend()
plt.show()
plt.close(f)
