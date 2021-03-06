import sys
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io
import os
import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels import api as sm


def pf(theta_deg):
    """Infer parameters for model fits."""
    def R_theta(t_deg, w1, w2, k):
        t_rad = t_deg / 180. * sp.pi
        theta_rad = theta_deg / 180. * sp.pi
        T0 = stats.vonmises.pdf(t_rad * 2., loc=0.0, scale=1.35, kappa=kappa)
        Tt = stats.vonmises.pdf(t_rad * 2., loc=theta_rad * 1.35, scale=1.35, kappa=kappa)  # noqa
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


# Prepare digitized data
gt_data = [
    [
        os.path.join("digitized_data", "mely", 'TB2015_%i_%s.csv' % (i, s))  # noqa
        for i in range(-90, 90, 30)] for s in ('PS', 'PO')
    ]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# Config
if len(sys.argv) == 2:
    file_name = sys.argv[1]
else:
    file_name = None
npoints = 128
markersize = 3
maxmarkersize = 10
linesize = 2
output_dir = "results_tb_fig4b"
os.makedirs(output_dir, exist_ok=True)

# get digitized data points
###########################
ps_x = np.zeros((6, 7))
ps_y = np.zeros((6, 7))
po_x = np.zeros((6, 7))
po_y = np.zeros((6, 7))
c2c1 = np.linspace(-90, 60, 6)

# load and wrap first point (x-wise) to last position
for idx, (csv_ps, csv_po) in enumerate(zip(*gt_data)):
    ps_x[idx, :-1], ps_y[idx, :-1] = \
        np.genfromtxt(csv_ps, delimiter=',').T
    po_x[idx, :-1], po_y[idx, :-1] = \
        np.genfromtxt(csv_po, delimiter=',').T
    ps_x[idx, -1] = 90
    po_x[idx, -1] = 90
    ps_y[idx, -1] = ps_y[idx, 0]
    po_y[idx, -1] = po_y[idx, 0]

# decoded mean vectors, digitized from the paper
do = np.array([16.67, -27.13, -12.57, 01.99, 22.50, 29.56])
ds = np.array([04.67, -11.25, 00.67, 06.61, 05.96, 10.51])

# fit: define parametric curve (Eq. 2, Trott & Born 2015)
#########################################################

# first estimate dispersion from special case
# (same ori, plaid-only; see paper for details) ...
i_nodiff = np.where(c2c1 != np.inf)[0][0]
_, _, kappa = curve_fit(R_0, xdata=po_x[i_nodiff], ydata=po_y[i_nodiff], maxfev=10**9)[0]  # noqa
po_par = np.zeros((6, 3))
ps_par = np.zeros((6, 3))
for theta, x_, y_, par in zip(c2c1, po_x, po_y, po_par):
    par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
for theta, x_, y_, par in zip(c2c1, ps_x, ps_y, ps_par):
    par[:], _ = curve_fit(pf(theta), xdata=x_, ydata=y_, maxfev=10**9)
po_fit = np.zeros((6, npoints))
ps_fit = np.zeros((6, npoints))
xx_rad = np.linspace(-np.pi, np.pi, npoints)
xx_deg = xx_rad / sp.pi * 90.
for idx, (theta, par, fit) in enumerate(zip(c2c1, po_par, po_fit)):
    fit[:] = pf(theta)(xx_deg, *par)
for idx, (theta, par, fit) in enumerate(zip(c2c1, ps_par, ps_fit)):
    fit[:] = pf(theta)(xx_deg, *par)

# Plot TB data
xticks = np.linspace(-90, 90, 7)

# Plot model data
no_surround = np.load('plaid_no_surround_outputs_data.npy')
surround = np.load('plaid_surround_outputs_data.npy')
no_surround = np.concatenate((no_surround, no_surround[0].reshape(1, -1)), 0)
surround = np.concatenate((surround, surround[0].reshape(1, -1)), 0)
images = 'plaid_surround/test/imgs/1'

stride = np.floor(180 / no_surround.shape[0]).astype(int)

thetas = {
    0: -90,
    30: -60,
    60: -30,
    90: 0,
    120: 30,
    150: 60,
    -1: 90,
}

idxs = np.asarray([int(x) for x in thetas.keys()])
ranges = np.asarray([int(x) for x in thetas.values()])

# Curve fits
surround_curve = surround[:, idxs.astype(int)].T
no_surround_curve = no_surround[:, idxs.astype(int)].T
i_nodiff = np.where(idxs != 90)[0]
xs = ranges.reshape(1, -1).repeat(len(surround_curve), 0).astype(float)
_, _, kappa = curve_fit(R_0, xdata=xs[i_nodiff].ravel(), ydata=no_surround_curve[i_nodiff].ravel(), maxfev=10**9)[0]  # noqa
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
f, axs = plt.subplots(3, 6, figsize=(8, 8))
max_surround, max_surround_idx = [], []
max_no_surround, max_no_surround_idx = [], []
plt.suptitle("Trott and Born Fig 3C")
f.text(0.5, 0.01, "Populations' preferred orientation", ha='center')
thetas = {k: v for k, v in list(thetas.items())[:-1]}
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Plot stimuli
for idx, (theta, label) in enumerate(thetas.items()):
    ax = axs[0, idx]
    it_image = io.imread(os.path.join(images, "sample_{}.png".format(theta)))  # noqa
    ax.imshow(it_image, cmap="Greys_r")
    ax.axis("off")

# T&B plots
# TODO: force x axis for mely extraction
for idx, (theta, label) in enumerate(thetas.items()):
    ax = axs[1, idx]
    ax.plot(
        po_x[idx],
        po_y[idx],
        color='#242424',
        alpha=.75,
        linestyle='None',
        markersize=markersize,
        marker='o')
    ax.plot(
        ps_x[idx],
        ps_y[idx],
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
    ax.set_ylim([0.0, max(po_y.max(), ps_y.max()) + 0.2])
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
        x=[do[idx]],
        color='#242424',
        linestyle="--",
        alpha=0.5)  # noqa
    ax.axvline(
        x=[ds[idx]],
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
    it_no_surround = no_surround[:, theta]
    it_surround = surround[:, theta]

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
        model_po_fit[idx],
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
        x=[model_xx_deg[np.argmax(model_po_fit[idx])]],
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
    ax.set_ylim([min(surround.min(), no_surround.min()) - 0.45, max(surround.max(), no_surround.max()) + 0.05])  # noqa
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
    plt.savefig(os.path.join(output_dir, "{}_model.pdf".format(file_name)))
else:
    plt.show()
plt.close(f)

# Prepare for stats
ns = no_surround_curve[:-1, :-1].reshape(-1, 1)
so = surround_curve[:-1, :-1].reshape(-1, 1)
model_data = np.concatenate((ns, so), 0)
# experiments = np.asarray([int(x) for x in thetas.keys()]).reshape(-1, 1).repeat(no_surround_curve.shape[0] - 1, 0)  # noqa
experiments = np.arange(
    surround_curve.shape[0] - 1).reshape(-1, 1).repeat(surround_curve.shape[0] - 1, -1).reshape(-1, 1)  # noqa
experiments = experiments.repeat(2, 1).T.reshape(-1, 1)
crf_ecrf = np.concatenate((np.zeros_like(ns), np.ones_like(so)), 0)
gt_ns = po_y[:, :-1].reshape(-1, 1)
gt_so = ps_y[:, :-1].reshape(-1, 1)
y = np.concatenate((gt_ns, gt_so), 0)
bias = np.ones_like(model_data)
X = np.concatenate((
    bias,
    model_data,
    experiments,
    crf_ecrf,), -1)
clf = sm.OLS(y, X).fit()
r2 = clf.rsquared
print("{} r^2: {}".format(r2))
np.save(os.path.join(output_dir, "{}_scores".format(file_name)), [r2, file_name])  # noqa
