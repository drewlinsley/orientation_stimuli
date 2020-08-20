import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import sys
from statsmodels import api as sm


# Config
linesize = 2
markersize = 9
if len(sys.argv) == 2:
    file_name = sys.argv[1]
else:
    file_name = None
plot_images = False
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

no_surround = np.load('contrast_modulated_no_surround_outputs_data.npy')
# no_surround = np.concatenate((no_surround, no_surround[0].reshape(1, -1)), 0)
metas = np.load("contrast_modulated_no_surround_outputs/1.npy", allow_pickle=True, encoding="latin1")  # noqa
metas = metas.reshape(-1, 14)
# mask = (np.asarray(metas[:, 4]).astype(int) == -45)
# metas = metas[mask]
# no_surround = no_surround[mask]
# surround = surround[mask]
contrasts = (metas[:, -2:]).astype(float)

# no_surround = no_surround.reshape(5, 5, 180)  # [..., 0]
# surround = surround.reshape(5, 5, 180)  # [..., 0]

# Roll so that they start at -45
no_surround = np.roll(no_surround, 180 - 135, axis=0)
# no_surround = np.concatenate((no_surround[0][None], no_surround), axis=0)

# idx3a = np.asarray([0, 5, 10, 20, 2, 7, 12, 17])

# print("Using center+Surround stim (fuller field).")
# no_surround = surround

# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# f = plt.figure(figsize=(7, 4))
# max_surround, max_surround_idx = [], []
# max_no_surround, max_no_surround_idx = [], []
# img_data = np.load("images_contrast_modulated_no_surround_outputs_data.npy")
# plt.suptitle("Busse and Wade Fig 3a")
# count = 1
# for c in range(4):
#     for r in range(2):
#         ax = plt.subplot(4, 4, count)
#         it_no_surround = no_surround[idx3a[count - 1]]
#         ax.plot(ranges, it_no_surround, label="no surround", color="Black", alpha=0.5, marker=".")  # noqa
#         plt.title(str(contrasts[count - 1]))
#         if r + c == 0:
#             plt.ylabel("Parameter loading")
#             # plt.legend(loc="best")
#         if r + c > 0:
#             # ax.set_yticklabels([""])
#             pass
#         if r == 4:
#             plt.xticks([-45, 0, 45, 90, 135])
#         else:
#             ax.set_xticklabels([""])
#         # plt.ylim([0.45, 0.6])

#         # Now load the image
#         ax = plt.subplot(4, 4, count + 8)
#         ax.imshow(img_data[idx3a[count - 8], ..., 0], cmap="Greys_r")
#         ax.axis("off")
#         count += 1
# plt.show()
# plt.close(f)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Plot paper data
ranges = np.linspace(-45, 135, y_paper.shape[-1])[:y_paper.shape[-1]]
f = plt.figure(figsize=(8, 5))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Busse, Wade, & Carandini Fig 4a")
count = 1
for r in range(5):
    for c in range(5):
        ax = plt.subplot(5, 5, count)
        it_neural = y_paper[r, c, :]
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
    "$\gamma$-net activity",  # noqa
    va="center",
    rotation="vertical")
if file_name is not None:
    plt.savefig(os.path.join(output_dir, "{}_model.pdf".format(file_name)))
else:
    plt.show()
plt.close(f)

# Plot images as a control
if plot_images:
    img_data = np.load("images_contrast_modulated_no_surround_outputs_data.npy")  # noqa
    plt.suptitle("Trott and Born Fig 3C")
    count = 1
    for r in range(5):
        for c in range(5):
            ax = plt.subplot(5, 5, count)
            ax.imshow(img_data[count - 1, ..., 0], cmap="Greys_r", vmin=img_data.min(), vmax=img_data.max())  # noqa
            plt.title(str(contrasts[count - 1]))
            ax.axis("off")
            count += 1
    if file_name is not None:
        plt.savefig(
            os.path.join(output_dir, "{}_images.pdf".format(file_name)))
    else:
        plt.show()
        plt.close(f)

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
    experiments.reshape(-1, 1),
), -1)
clf = sm.OLS(y, X).fit()
r2 = clf.rsquared
print("{} r^2: {}".format(r2))
np.save(
    os.path.join(output_dir, "{}_scores".format(file_name)), [r2, file_name])
