import os
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches


mdict = {
"full": "results_full",
"multiplicative only": "results_gammanet_bsds_multiplicativeonly",
"horizontal only": "results_gammanet_bsds_honly",
"additive only": "results_gammanet_bsds_additiveonly",
"top-down only": "results_gammanet_bsds_tdonly",
"feedforward only": "results_gammanet_bsds_ffonly",
"eCRF-Large": "results_gammanet_bsds_ecrf_plus",
"eCRF-Small": "results_gammanet_bsds_ecrf",
"CRF": "results_gammanet_bsds_crf"
}

data_dir = "/media/data_cifs/projects/prj_neural_circuits/py-bsds500"
perfs = {}
for k, v in mdict.items():
    dirs = glob(os.path.join(data_dir, v, "*"))
    iperf = []
    for d in dirs:
        try:
            iperf.append(np.load(d)["best_f1"])
        except:
            print("Failed on {}".format(d))
    perfs[k] = np.max(iperf)
    print("{} max is {}".format(k, dirs[np.argmax(iperf)]))

# Human data
human = np.load("../py-bsds500/human_held_out_reliability.npy")
low_human = human.min()
high_human = human.max()

# Create df
df = pd.DataFrame(np.concatenate((np.asarray([x for x in perfs.keys()]).reshape(-1, 1), np.asarray([x for x in perfs.values()]).reshape(-1, 1), np.arange(len(perfs.keys())).reshape(-1, 1)), -1), columns=["Model", "ODS F1 Score", "idx"])
df["ODS F1 Score"] = pd.to_numeric(df["ODS F1 Score"])
df["idx"] = pd.to_numeric(df["idx"])

# Split into data/loss manipulation and model manipulation sets
dataloss_set = ["full", "CRF", "eCRF-Small", "eCRF-Large"]
modellesion_set = ["multiplicative only", "horizontal only", "additive only", "top-down only", "feedforward only"]
dataloss_df = df[np.isin(df.Model.values, dataloss_set)]  # .reset_index()
modellesion_df = df[np.isin(df.Model.values, modellesion_set)]  # .reset_index()
dataloss_df = dataloss_df.reset_index(drop=True)
modellesion_df = modellesion_df.reset_index(drop=True)
dataloss_df.idx = np.arange(len(dataloss_df))
modellesion_df.idx = np.arange(len(modellesion_df))

# Plot dataloss
fig, ax = plt.subplots()
order = np.argsort(dataloss_df["ODS F1 Score"].values)[::-1]
g = sns.barplot(x="idx", y="ODS F1 Score", data=dataloss_df, palette="husl", order=order)
# df.sort_values("ODS F1 Score", ascending=False).index)
# ax.get_legend().remove()
ax.set_xticklabels(np.asarray([x for x in dataloss_df.Model.values])[order], rotation_mode='anchor', ha="right")
# ax.set_xlim([0.6, 0.9])
ax.set_ylim([0.6, 0.9])
ax.add_patch(
    patches.Rectangle(
        (-1, low_human),
        len(dataloss_df) + 1,
        high_human - low_human,
        alpha=0.4,
        zorder=-100,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Plot modellesion
fig, ax = plt.subplots()
order = np.argsort(modellesion_df["ODS F1 Score"].values)[::-1]
g = sns.barplot(x="idx", y="ODS F1 Score", data=modellesion_df, palette="Set2", order=order)
# df.sort_values("ODS F1 Score", ascending=False).index)
# ax.get_legend().remove()
ax.set_xticklabels(np.asarray([x for x in modellesion_df.Model.values])[order], rotation_mode='anchor', ha="right")
# ax.set_xlim([0.6, 0.9])
ax.set_ylim([0.6, 0.9])
ax.add_patch(
    patches.Rectangle(
        (-1, low_human),
        len(modellesion_df) + 1,
        high_human - low_human,
        alpha=0.4,
        zorder=-100,
        linewidth=0,
        hatch='///',
        facecolor='grey'))

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

