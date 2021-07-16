import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from glob import glob


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()
    # cmap = sns.cubehelix_palette(100, light=0.7)
    # cmap = sns.color_palette("Reds", 100)
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        # color = cmap[int(w * 100)]
        # rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
        #                      facecolor=color, edgecolor=color)
        rect = plt.Circle((x, y), radius=size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


file_dirs = [
    # "results_orientation_tuning",
    # "results_tb_fig4b_center",
    # "results_tb_fig4b_surround",
    # "results_tb_fig4b_full",
    # "results_tb_fig2_full",
    "results_tb_fig4b_diff",
    "results_tb_fig2_diff",
    "results_bwc_fig4a",
]
remap_models = {
    "simple_no_additive": "Normalization circuit",
    "simple": "Complete",
    "simple_crf": "CRF patch-trained",
    "simple_ecrf": "eCRF patch-trained",  # $\gamma$-net",
    "simple_ecrf_bigger": "2xeCRF patch-trained",  #  $\gamma$-net",
    "simple_no_nonnegative": "No non-negative constraint",
    "simple_ts_1": "Complete speeded",
    "horizontal": "H-connections",
    "horizontal_bigger": "Wide H-connections",
    "simple_no_additive_no_multiplicative": "+ and * lesion",  # noqa Remove this for now
    "simple_no_h": "TD-connections",
    "simple_untied": "Feedforward only",
    "simple_no_multiplicative": "Bias circuit"
}
remap_experiments = {
    # "results_orientation_tuning": "Orientation decoding",
    # "results_tb_fig4b_center": "T&B 2015: CRF tuning",
    # "results_tb_fig4b_surround": "T&B 2015: eCRF tuning",
    "results_tb_fig4b_full": "T&B 2015: CRF+eCRF FS supp",
    "results_tb_fig4b_diff": "Selective gain control", # "T&B 2015: Feature-Selective eCRF suppression",
    # "results_tb_fig2_full": "T&B 2015: CRF+eCRF supp",
    "results_tb_fig2_diff": "Tuned gain control",  # "T&B 2015: Tuned eCRF suppression",
    "results_bwc_fig4a": "BWC 2013: WTA CRF",
    "Mean $R^2$ performance": "Mean Neural $R^2$",
    # "bsds_5": "BSDS $5\%$-data",
    # "bsds_10": "BSDS $10\%$-data",
    # "bsds_100": "BSDS $100\%$-data",
}

remove_models = [
    "gammanet_full",
    "simple_no_additive_no_multiplicative",
    "simple_untied",
    "horizontal_bigger",
    "simple_no_nonnegative",
    "simple_ts_1",

    # "simple_crf",  # : "CRF patch-trained",
    # "simple_ecrf",  # : "eCRF patch-trained",  # $\gamma$-net",
    # "simple_ecrf_bigger",  # : "+eCRF patch-trained",  #  $\gamma$-net",
]

exp_order = [
    'results_bwc_fig4a',
    'results_tb_fig2_diff',  # Use this to describe tuning of surrounds?
    # 'results_tb_fig4b_diff',  # Only compare WTA in CRF vs. eCRF
    'Mean $R^2$ performance',
    # 'bsds_5',
    # 'bsds_10',
    # 'bsds_100'
]

bsds_file = "bsds_results.csv"
bsds_data = pd.read_csv(bsds_file)
wildcard = "INSILICO_*"
data, models, experiments = [], [], []
for d in file_dirs:
    files = glob(os.path.join(d, "*.npy"))
    for f in files:
        i = np.load(f)
        name = i[1].replace("INSILICO_BSDS_vgg_gratings_", "")
        if name in set(remap_models.keys()):
            data.append(float(i[0]))
            models.append(name)
            experiments.append(d)  # noqa
        else:
            print("Excluding {}".format(f))
data = np.asarray(data)
models = np.asarray(models)
experiments = np.asarray(experiments)

# Package into a DF
df = pd.DataFrame(
    np.stack((data, models, experiments), -1),
    columns=["data", "models", "experiments"])

# Concat extra data
df = pd.concat((df, bsds_data), 0)

# Remove any models we dont need
for m in remove_models:
    df = df[df.models != m]

# Transform long -> wide
wdf = df.pivot(index="experiments", columns="models", values="data")
wdf = wdf.reset_index()
wdf_all = wdf.reset_index().values
experiment_names = wdf_all[:, 1]
model_names = wdf.columns.values
wdf_data = wdf_all[:, 2:].astype(np.float32)

# Add mean performance
wdf_data_score = np.nanmean(wdf_data, 0)
if np.any(np.isnan(wdf_data)):
    print("WARNING: NaNs found. Is there missing data?")
model_order = np.argsort(wdf_data_score)[::-1]
wdf_data = np.concatenate((wdf_data, [wdf_data_score]), 0)
experiment_names = np.concatenate((experiment_names, ["Mean $R^2$ performance"]))  # noqa

# Organize experiments
idx = np.asarray(np.asarray([experiment_names.tolist().index(ex) for ex in exp_order]))  # noqa
wdf_data = wdf_data[idx]  # Re-order experiments
experiment_names = experiment_names[idx]

# Prep plots
sorted_wdf_data = wdf_data[:, model_order]
sorted_models = model_names[1:][model_order]
fixed_models = []
for m in sorted_models:
    for k, v in remap_models.items():
        if m == k:
            fixed_models.append(v)
fixed_models = np.asarray(fixed_models)
fixed_exps = []
for m in experiment_names:
    for k, v in remap_experiments.items():
        if m == k:
            fixed_exps.append(v)
fixed_exps = np.asarray(fixed_exps)

# Plot with a heatmap
f, ax = plt.subplots(1, 1, dpi=150)
plt.title("Model fits to neural and perceptual data")
g = sns.heatmap(
    data=sorted_wdf_data.T,
    square=True,
    cmap="RdYlBu_r",  # "coolwarm",  # "flare_r",  # "YlGnBu",
    vmin=0.,
    vmax=0.8,
    cbar_kws={"label": "$R^2$"})
g.set_yticklabels(fixed_models, size=8)
g.set_xticklabels(fixed_exps, size=8, ha="right")
plt.yticks(rotation=0)
plt.xticks(rotation=30)
# plt.ylabel("Models")
plt.xlabel("Experiments")
plt.tight_layout()
plt.show()
plt.close(f)

# Plot barplots
df = pd.DataFrame(sorted_wdf_data.T[::-1, :-1][:, ::-1], columns=['eCRF', 'CRF'])
# df = pd.DataFrame(sorted_wdf_data.T[:, :-1], columns=['CRF', 'eCRF'])

axes = df.plot.barh(color={"#007535", "#97ada1"}, figsize=(5, 7))
# axes = df.plot.bar(color={"#007535", "#97ada1"}, figsize=(5, 7))
axes.legend(bbox_to_anchor=(0, 1), loc='center right', ncol=1)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
# plt.tight_layout()
plt.show()
plt.close("all")

"""
# Plot with a hinton
f, ax = plt.subplots(1, 1, dpi=150)
plt.title("Model fits to neural and perceptual data")
hinton(sorted_wdf_data, ax=ax, max_weight=4)
ax.set_yticks(range(len(fixed_models)))
ax.set_xticks(range(len(fixed_exps)))
ax.set_yticklabels(fixed_models, size=8)
ax.set_xticklabels(fixed_exps, size=8, ha="right")
plt.yticks(rotation=0)
plt.xticks(rotation=30)
# plt.ylabel("Models")
plt.xlabel("Experiments")
plt.tight_layout()
plt.show()
plt.close(f)
"""

####
