import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import sys
from statsmodels import api as sm
from skimage import io


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


# Config
linesize = 2
markersize = 9
if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = None

if len(sys.argv) > 2:
    no_surround = np.load(sys.argv[2])
else:
    no_surround = np.load('contrast_modulated_no_surround_outputs_data.npy')

plot_images = True
output_dir = "results_bwc_fig4a"
os.makedirs(output_dir, exist_ok=True)

# retrieve digitized data
v_contrasts = [0.0, .06, .12, .25, .50]
h_contrasts = [0.0, .06, .12, .25, .50]
nv, nh = len(v_contrasts), len(h_contrasts)
csv_cts = np.asarray([0.0, .06, .12, .25, .50]) * 100
csvfiles = np.asarray([[
    os.path.join(
        "digitized_data",
        "mely",
        'BWC2009_%i_%i.csv' % (i, j)) for i in csv_cts] for j in csv_cts]).T
t_paper = np.zeros((nv, nh, 13))
y_paper = np.zeros((nv, nh, 13))
for idx in range(nv):
    for jdx in range(nh):
        t_paper[idx, jdx], y_paper[idx, jdx] = np.genfromtxt(
            csvfiles[idx, jdx], delimiter=',').T

# no_surround = np.load('contrast_modulated_no_surround_outputs_data.npy')
# no_surround = np.concatenate((no_surround, no_surround[0].reshape(1, -1)), 0)
metas = np.load("contrast_modulated_no_surround/test/metadata/1.npy", allow_pickle=True, encoding="latin1")  # noqa
metas = metas.reshape(-1, 14)
# mask = (np.asarray(metas[:, 4]).astype(int) == -45)
# metas = metas[mask]
# no_surround = no_surround[mask]
# surround = surround[mask]
contrasts = (metas[:, -2:]).astype(float)

# no_surround = no_surround.reshape(5, 5, 180)  # [..., 0]
# surround = surround.reshape(5, 5, 180)  # [..., 0]

# # Roll so that they start at -45
no_surround = np.roll(no_surround, 180 - 135, axis=0)
# no_surround = np.roll(no_surround, 3, axis=0)

# no_surround = np.concatenate((no_surround[0][None], no_surround), axis=0)

# print("Using center+Surround stim (fuller field).")
# no_surround = surround


sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Plot paper data
ranges = np.linspace(-45, 135, y_paper.shape[-1])[:y_paper.shape[-1]]
f = plt.figure(figsize=(8, 5))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Busse, Wade, & Carandini Fig 4a")
count = 1
neural_diffs = []
for r in range(5):
    for c in range(5):
        ax = plt.subplot(5, 5, count)
        it_neural = y_paper[r, c, :]
        neural_diffs.append([it_neural[3], it_neural[9]])
        ax.plot(
            ranges,
            it_neural,
            label="no surround",
            color="Black",
            alpha=0.5,
            linestyle='-',
            linewidth=linesize,
            markersize=markersize,
            marker=".")  # noqa
        # plt.title(str(contrasts[count - 1]))
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlim([-50, 150])
        ax.set_xticks([-45, 0, 45, 90, 135])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
        if r != 4:
            ax.set_xticklabels([""])
        if c != 0:
            ax.set_yticklabels([""])
        count += 1

        # plt.ylim([-0.05, 0.1])
f.text(
    0.04,
    0.5,
    "Neural population response",
    va="center",
    rotation="vertical")
if file_name is not None:
    plt.savefig(os.path.join(output_dir, "{}_neural.pdf".format(file_name)))
else:
    plt.show()
plt.close(f)

# Plot model data
ranges = np.linspace(-45, 135, no_surround.shape[0] + 1)[:no_surround.shape[0] + 1]  # noqa
f = plt.figure(figsize=(8, 5))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Busse, Wade, & Carandini Fig 4a")
count = 1
for r in range(5):
    for c in range(5):
        ax = plt.subplot(5, 5, count)
        it_no_surround = no_surround[:, count - 1]
        it_no_surround = np.concatenate((it_no_surround, [it_no_surround[0]]), 0)  # noqa
        ax.plot(
            ranges,
            it_no_surround,
            label="no surround",
            color="Black",
            alpha=0.5,
            linestyle='-',
            linewidth=linesize,
            markersize=markersize,
            marker=".")  # noqa
        ax.set_ylim([-0.1, 0.8])  # 0.5
        # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlim([-50, 150])
        ax.set_xticks([-45, 0, 45, 90, 135])
        if r != 4:
            ax.set_xticklabels([""])
        if c != 0:
            ax.set_yticklabels([""])
        count += 1

        # plt.ylim([-0.05, 0.1])
f.text(
    0.04,
    0.5,
    "$\gamma$-net activity",  # noqa
    va="center",
    rotation="vertical")
if file_name is not None:
    plt.savefig(os.path.join(output_dir, "{}_model.pdf".format(file_name)))
else:
    plt.show()
plt.show()
plt.close(f)

# plt.imshow(no_surround)
# plt.show()

os._exit(1)


# Prepare for stats
no_surround_stats = no_surround.T
y_paper_stats = y_paper.reshape(25, 13)[:, :-1]  # noqa omit 135, which = -45 (0th element)
experiments = np.arange(
    y_paper_stats.shape[1]).reshape(-1, 1).repeat(y_paper_stats.shape[0], -1)
y = y_paper_stats.reshape(-1, 1)
bias = np.ones_like(y)
X = np.concatenate((
    bias,
    no_surround_stats.reshape(-1, 1),
), -1)
clf = sm.OLS(y, X).fit()
r2 = clf.rsquared
# print("BWC {} r^2: {}".format(file_name, r2))
np.save(
    os.path.join(output_dir, "{}_scores".format(file_name)), [r2, file_name])

# Per-panel states
so = no_surround_stats.reshape(-1, 1)  # surround_curve[:-1, :-1].reshape(-1, 1)
gt_so = y_paper_stats.reshape(-1, 1)  # ps_y[:, :-1].reshape(-1, 1)
r2s = []
inds = np.arange(25).repeat(12)
for idx in np.arange(25):
    it_mod = so[inds == idx]
    it_gt = gt_so[inds == idx]
    bias = np.ones_like(it_mod)
    X = np.concatenate((
        bias,
        it_mod,), -1)
    clf = sm.OLS(it_gt, X).fit()
    r2 = clf.rsquared
    r2s.append(r2)
r2 = np.mean(r2s)
print("BWC {} r^2: {}".format(file_name, r2))
np.save(
    os.path.join(output_dir, "{}_diff_scores".format(file_name)), [r2, file_name])

