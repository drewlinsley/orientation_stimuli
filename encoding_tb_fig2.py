import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels import api as sm
import pandas as pd
import sys
from skimage import io


def pf(theta_deg):
    """Infer parameters for model fits."""
    def R_theta(t_deg, w1, w2, k):
        t_rad = t_deg / 180. * sp.pi
        theta_rad = theta_deg / 180. * sp.pi
        T0 = stats.vonmises.pdf(t_rad * 2., loc=0.0, scale=1., kappa=kappa)
        Tt = stats.vonmises.pdf(t_rad * 2., loc=theta_rad * 2., scale=1., kappa=kappa)  # noqa
        T0 /= T0.max()
        Tt /= Tt.max()
        return w1 * T0 + w2 * Tt - k
    return R_theta


def R_0(t_deg, w, k, kappa):
    """PDF for TB2015 model fits."""
    t_rad = t_deg / 180. * sp.pi
    T0 = stats.vonmises.pdf(t_rad * 2., loc=0.0, scale=1., kappa=kappa)
    T0 /= T0.max()
    return w * T0 - k


# Config
images = 'orientation_tilt/test/imgs/1'
npoints = 128
markersize = 3
maxmarkersize = 10
linesize = 2
if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = None
if len(sys.argv) > 2:
    no_surround = np.load(sys.argv[2])
    surround = np.load(sys.argv[3])
else:
    no_surround = np.load('plaid_no_surround_outputs_data.npy')
    surround = np.load('orientation_tilt_outputs_data.npy')

xticks = np.linspace(-90, 90, 7)
output_dir_full = "results_tb_fig2_full"
output_dir_diff = "results_tb_fig2_diff"
os.makedirs(output_dir_full, exist_ok=True)
os.makedirs(output_dir_diff, exist_ok=True)

# Load TB data
neural_surround= pd.read_csv("digitized_data/drew/tb_fig2_surround.csv", header=None).values[:, 1]  # noqa
neural_no_surround= pd.read_csv("digitized_data/drew/tb_fig2_no_surround.csv", header=None).values[:, 1]  # noqa
neural_surround = neural_surround.reshape(6, -1).T
neural_no_surround = neural_no_surround.reshape(6, -1).T
neural_surround = np.concatenate((neural_surround, neural_surround[0].reshape(1, -1)))  # noqa
neural_no_surround = np.concatenate((neural_no_surround, neural_no_surround[0].reshape(1, -1)))  # noqa
neural_surround = neural_surround.T
neural_no_surround = neural_no_surround.T
c2c1 = np.linspace(-90, 60, 6)

# Fix the flip and scale from digital extraction
neural_no_surround = neural_no_surround.max() - neural_no_surround
neural_surround = neural_surround.max() - neural_surround
neural_no_surround = (neural_no_surround - neural_no_surround.min()) / neural_no_surround.max()  # noqa
neural_no_surround = (1.5 - 0.65) * neural_no_surround + 0.65
neural_surround = (neural_surround - neural_surround.min()) / neural_surround.max()  # noqa
neural_surround = (1.25 - 0.5) * neural_surround + 0.5

# Load model data
no_surround = np.concatenate((no_surround, no_surround[0].reshape(1, -1)))
surround = np.concatenate((surround, surround[0].reshape(1, -1)))

# Get TB model fits
# fit: define parametric curve (Eq. 2, Trott & Born 2015)
#########################################################

# first estimate dispersion from special case
# (same ori, plaid-only; see paper for details) ...
thetas = {
    1: -90,
    31: -60,
    61: -30,
    91: 0,
    121: 30,
    151: 60,
    -1: 90,
}

ranges = np.asarray([int(x) for x in thetas.values()])
xs = ranges.reshape(1, -1).repeat(len(neural_no_surround), 0).astype(float)

i_nodiff = np.where(xs[0] == 0)[0]
_, _, kappa = curve_fit(R_0, xdata=xs[i_nodiff].flatten(), ydata=neural_no_surround[i_nodiff].flatten(), maxfev=10**9)[0]  # noqa
po_par = np.zeros((6, 3))
ps_par = np.zeros((6, 3))
for theta, x_, y_, par in zip(c2c1, xs, neural_no_surround, po_par):
    par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
for theta, x_, y_, par in zip(c2c1, xs, neural_surround, ps_par):
    par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
po_fit = np.zeros((6, npoints))
ps_fit = np.zeros((6, npoints))
xx_rad = np.linspace(-np.pi, np.pi, npoints)
xx_deg = xx_rad / np.pi * 90.
for idx, (theta, par, fit) in enumerate(zip(c2c1, po_par, po_fit)):
    fit[:] = pf(theta)(xx_deg, *par)
for idx, (theta, par, fit) in enumerate(zip(c2c1, ps_par, ps_fit)):
    fit[:] = pf(theta)(xx_deg, *par)

# Get gammanet model fits
stride = np.floor(180 / no_surround.shape[0]).astype(int)
thetas = {
    1: -90,
    31: -60,
    61: -30,
    91: 0,
    121: 30,
    151: 60,
    -1: 90,
}

idxs = np.asarray([int(x) for x in thetas.keys()])[:-1]
ranges = np.asarray([int(x) for x in thetas.values()])
# surround_curve = surround[:, idxs.astype(int)].T
# no_surround_curve = no_surround[:, idxs.astype(int)].T

# Curve fits
surround_curve = surround[:, idxs.astype(int)].T
no_surround_curve = no_surround[:, idxs.astype(int)].T
i_nodiff = np.where(xs[0] == 0)[0]
xs = ranges.reshape(1, -1).repeat(len(surround_curve), 0).astype(float)
_, _, kappa = curve_fit(R_0, xdata=xs[i_nodiff].flatten(), ydata=no_surround_curve[i_nodiff].flatten(), maxfev=10**9)[0]  # noqa
po_par = np.zeros((len(surround_curve), 3))
ps_par = np.zeros((len(surround_curve), 3))
for theta, x_, y_, par in zip(ranges, xs, no_surround_curve, po_par):
    par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
for theta, x_, y_, par in zip(ranges, xs, surround_curve, ps_par):
    par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
model_po_fit = np.zeros((len(surround_curve), npoints))
model_ps_fit = np.zeros((len(surround_curve), npoints))
xx_rad = np.linspace(-np.pi, np.pi, npoints)
model_xx_deg = xx_rad / sp.pi * 90.
for idx, (theta, par, fit) in enumerate(zip(c2c1, po_par, model_po_fit)):
    fit[:] = pf(theta)(xx_deg, *par)
for idx, (theta, par, fit) in enumerate(zip(c2c1, ps_par, model_ps_fit)):
    fit[:] = pf(theta)(xx_deg, *par)

# Prepare figure
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
f, axs = plt.subplots(3, 6, figsize=(8, 8))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Trott and Born Fig 2")
f.text(0.5, 0.01, "Populations' preferred orientation", ha='center')
thetas = {k: v for k, v in list(thetas.items())[:-1]}

# Plot stimuli
for idx, (theta, label) in enumerate(thetas.items()):
    ax = axs[0, idx]
    it_image = io.imread(os.path.join(images, "sample_{}.png".format(theta)))  # noqa
    ax.imshow(it_image[150:-150], cmap="Greys_r")
    ax.axis("off")

# T&B plots
# TODO: force x axis for mely extraction
for idx, (theta, label) in enumerate(thetas.items()):
    ax = axs[1, idx]
    ax.plot(
        xs[idx],
        neural_no_surround[idx],
        color='#242424',
        alpha=.75,
        linestyle='None',
        markersize=markersize,
        marker='o')
    ax.plot(
        xs[idx],
        neural_surround[idx],
        color='#E03412',
        alpha=.75,
        linestyle='None',
        markersize=markersize,
        marker='o')

    # fits
    ax.plot(
        xx_deg,
        po_fit[idx],
        color='#242424',
        alpha=.50,
        markersize=0,
        label='Center only',
        linestyle='-',
        linewidth=linesize)
    ax.plot(
        xx_deg,
        ps_fit[idx],
        color='#E03412',
        alpha=.50,
        markersize=0,
        label='Center only',
        linestyle='-',
        linewidth=linesize)

    ax.set_xlim([-95, 95])
    ax.set_ylim([0.0, max(neural_no_surround.max(), neural_surround.max()) + 0.2])  # noqa
    ax.set_xticks(xticks)
    ax.set_xticklabels([""])

    # # markers
    # ax.plot(
    #     [do[idx]],
    #     [ax.get_ylim()[0] + 0.05],
    #     marker='^',
    #     color='#242424',
    #     alpha=0.75,
    #     markersize=maxmarkersize)
    # ax.plot(
    #     [ds[idx]],
    #     [ax.get_ylim()[0] + 0.05],
    #     marker='^',
    #     color='#E03412',
    #     alpha=0.75,
    #     markersize=maxmarkersize)
    # Draw verticle lines
    ax.axvline(
        x=[model_xx_deg[np.argmax(po_fit[idx])]],
        color='#242424',
        linestyle="--",
        alpha=0.5)  # noqa
    ax.axvline(
        x=[model_xx_deg[np.argmax(ps_fit[idx])]],
        color='#E03412',
        linestyle="--",
        alpha=0.5)  # noqa
    if idx > 0:
        ax.set_yticklabels([""])
    if idx == 0:
        ax.set_ylabel("Neural activity")

# Model Plots
for idx, (theta, label) in enumerate(thetas.items()):
    theta = int(theta)
    ax = axs[2, idx]
    it_surround = surround[:, theta]
    it_no_surround = no_surround[:, 90]
    it_model_no_surround = model_po_fit[3]

    # points
    ax.plot(
        xs[idx],
        it_no_surround,
        color='#242424',
        alpha=.75,
        linestyle='None',
        markersize=markersize,
        marker='o')
    ax.plot(
        xs[idx],
        it_surround,
        color='#E03412',
        alpha=.75,
        linestyle='None',
        markersize=markersize,
        marker='o')

    # fits
    ax.plot(
        model_xx_deg,
        it_model_no_surround,
        color='#242424',
        alpha=.50,
        markersize=0,
        label='Center only',
        linestyle='-',
        linewidth=linesize)
    ax.plot(
        model_xx_deg,
        model_ps_fit[idx],
        color='#E03412',
        alpha=.50,
        markersize=0,
        label='Center only',
        linestyle='-',
        linewidth=linesize)

    # # max-markers
    # ax.plot(
    #     [model_xx_deg[np.argmax(model_po_fit[idx])]],
    #     [min(surround.min(), no_surround.min()) - 0.05],
    #     marker='^',
    #     color='#242424',
    #     alpha=0.75,
    #     markersize=maxmarkersize)
    # ax.plot(
    #     [model_xx_deg[np.argmax(model_ps_fit[idx])]],
    #     [min(surround.min(), no_surround.min()) - 0.05],
    #     marker='^',
    #     color='#E03412',
    #     alpha=0.75,
    #     markersize=maxmarkersize)

    # Draw verticle lines
    ax.axvline(
        x=[model_xx_deg[np.argmax(it_model_no_surround)]],
        color='#242424',
        linestyle="--",
        alpha=0.5)  # noqa
    ax.axvline(
        x=[model_xx_deg[np.argmax(model_ps_fit[idx])]],
        color='#E03412',
        linestyle="--",
        alpha=0.5)  # noqa

    # Set axes
    ax.set_xlim([-95, 95])
    ax.set_ylim([min(surround.min(), no_surround.min()) - 0.45, max(surround.max(), no_surround.max()) + 0.0])  # noqa
    ax.set_xticks(xticks)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        # specify integer or one of preset strings, e.g.
        # tick.label.set_rotation('vertical')
    for label in ax.get_xticklabels():
        label.set_rotation(-45)

    if idx == 0:
        ax.set_ylabel("$\gamma$-net activity")  # noqa
    #     # plt.legend(loc="best")
    if idx > 0:
        ax.set_yticklabels([""])
if file_name is not None:
    plt.savefig(os.path.join(output_dir_full, "{}_model.pdf".format(file_name)))
else:
    plt.show()
plt.show()
plt.close(f)

# # Stats
ns = no_surround_curve[3, :-1].reshape(-1, 1).T.repeat(6, axis=0).reshape(-1, 1)  # noqa
so = surround_curve[:, :-1].reshape(-1, 1)
gt_ns = neural_no_surround[3, :-1].reshape(-1, 1).T.repeat(6, axis=0).reshape(-1, 1)
gt_so = neural_surround[:, :-1].reshape(-1, 1)

# Get the difference between args of the model fits
arg_diff_no_surround = [90] * 6  # This is fixed # np.argmax(model_po_fit[:-1], -1)
arg_diff_surround = np.argmax(model_ps_fit, -1)
arg_diff = arg_diff_no_surround - arg_diff_surround
# Get the difference between magnitude of the args
arg_mags_no_surround = np.asarray([x[y] for x, y in zip(model_po_fit, arg_diff_no_surround)])
arg_mags_surround = np.asarray([x[y] for x, y in zip(model_ps_fit, arg_diff_surround)])
arg_mags = arg_mags_no_surround - arg_mags_surround
# Concat diffs and mags into array for linear model
# diffs = np.concatenate((arg_diff_no_surround, arg_diff_surround), 0)
# mags = np.concatenate((arg_mags_no_surround, arg_mags_surround), 0)
# X = np.stack((diffs, mags), -1).reshape(-1, 1)
# X = np.stack((arg_diff, arg_mags), -1).reshape(-1, 1)
X = arg_mags.reshape(-1, 1)
X = np.concatenate((np.ones((len(X), 1)), X), -1)

# Do the same for gt data
gt_arg_diff_no_surround = np.argmax(po_fit, -1)
gt_arg_diff_surround = np.argmax(ps_fit, -1)
gt_arg_diff = gt_arg_diff_no_surround - gt_arg_diff_surround
gt_arg_mags_no_surround = np.asarray([x[y] for x, y in zip(po_fit, gt_arg_diff_no_surround)])
gt_arg_mags_surround = np.asarray([x[y] for x, y in zip(ps_fit, gt_arg_diff_surround)])
gt_arg_mags = gt_arg_mags_no_surround - gt_arg_mags_surround
# diffs = np.concatenate((gt_arg_diff_no_surround, gt_arg_diff_surround), 0)
# mags = np.concatenate((gt_arg_mags_no_surround, gt_arg_mags_surround), 0)
# y = np.stack((diffs, mags), -1).reshape(-1, 1)
# y = np.stack((arg_diff, arg_mags), -1).reshape(-1, 1)
y = arg_mags.reshape(-1, 1)

# Fit model
clf = sm.OLS(y, X).fit()
r2 = clf.rsquared
# THIS IS THE DEFAULT SCORE
# print("Arg score {}".format(r2))
np.save(os.path.join(output_dir_full, "{}_full_scores".format(file_name)), [r2, file_name])  # noqa

# compare full curve fits
y = np.concatenate((po_fit.ravel(), ps_fit.ravel()), 0)
X = np.concatenate((model_po_fit.ravel(), model_ps_fit.ravel()), 0)
X = np.stack((np.ones((len(X))), X), -1)
clf = sm.OLS(y, X).fit()
r2 = clf.rsquared
# THIS IS THE DEFAULT SCORE
print("Default score: {}".format(r2))
np.save(os.path.join(output_dir_full, "{}_full_scores".format(file_name)), [r2, file_name])  # noqa

# Fit to cat(c/c+s)
model_data = np.concatenate((ns, so), 0)
experiments = np.arange(
    surround_curve.shape[0]).reshape(-1, 1).repeat(surround_curve.shape[0], -1).reshape(-1, 1)  # noqa
# experiments = np.asarray([int(x) for x in thetas.keys()]).reshape(-1, 1).repeat(no_surround_curve.shape[1] - 1, 0)  # noqa
experiments = experiments.repeat(2, 1).T.reshape(-1, 1)
crf_ecrf = np.concatenate((np.zeros_like(ns), np.ones_like(so)), 0)
y = np.concatenate((gt_ns, gt_so), 0)
bias = np.ones_like(model_data)
X = np.concatenate((
    bias,
    model_data,
    crf_ecrf,), -1)
clf = sm.OLS(y, X).fit()
r2 = clf.rsquared
# print("TB surround sup FULL {} r^2: {}".format(file_name, r2))
np.save(os.path.join(output_dir_full, "{}_scores".format(file_name)), [r2, file_name])  # noqa

# Fit to Diff
so = surround_curve[:, :-1].reshape(-1, 1)
gt_so = neural_surround[:, :-1].reshape(-1, 1)
r2s = []
inds = np.arange(6).repeat(6)
for idx in np.arange(6):
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
# print("TB surround sup diff {} r^2: {}".format(file_name, r2))
np.save(os.path.join(output_dir_diff, "{}_full_scores".format(file_name)), [r2, file_name])  # noqa

# Make a histogram of the c/c+s differences
model_data = ns[np.arange(3, len(ns), 6)] - so[np.arange(3, len(ns), 6)]
model_data = (model_data - model_data.min(0)) / model_data.max(0)
y = gt_ns[np.arange(3, len(ns), 6)] - gt_so[np.arange(3, len(ns), 6)]
model_data = ns[np.arange(3, len(ns), 6)] - so[np.arange(3, len(ns), 6)]
y = (y - y.min(0)) / y.max(0)
f, ax = plt.subplots(1, 1, dpi=75)
plt.plot(model_data, "ko", alpha=0.6)
plt.title("Model data")
ax.set_ylim([-0.05, 1.05])
plt.xlabel("Orientation")
plt.ylabel("90-response diff")
plt.savefig(os.path.join(output_dir_full, "{}_c_cs_model_diff.pdf".format(file_name)))  # noqa
# plt.show()
plt.close(f)

f, ax = plt.subplots(1, 1, dpi=75)
plt.plot(y, "ko", alpha=0.6)
plt.title("Neural data")
ax.set_ylim([-0.05, 1.05])
plt.xlabel("Orientation")
plt.ylabel("90-response diff")
plt.savefig(os.path.join(output_dir_full, "{}_c_cs_neural_diff.pdf".format(file_name)))  # noqa
# plt.show()
plt.close(f)
