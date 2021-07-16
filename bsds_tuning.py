import os
import numpy as np
from glob import glob
from joblib import load
from matplotlib import pyplot as plt


path = "/Users/drewlinsley/Downloads/bsds_enc/"
inv = np.linalg.inv
files = glob(os.path.join(path, "*.npz"))
# files = [files[0]]
files = [files[14]]
height, width = 4, 4
clf = load("linear_models/tb_model.joblib")
moments = np.load("linear_moments/tb_feature_matrix.npz", allow_pickle=True)
# clf = load("linear_models/conv2_2_tb_model.joblib")
# moments = np.load("linear_models/conv2_2_tb_feature_matrix.npz", allow_pickle=True)
normalize = False
means = moments["means"]
stds = moments["stds"]
for fi, f in enumerate(files):
    d = np.load(f)
    im = d["im"].squeeze()
    act = d["fgru"].squeeze()
    argmaxes = np.zeros((act.shape[0], act.shape[1]), np.float32)
    sel_def = np.zeros_like(argmaxes)
    tcs = np.zeros((act.shape[0], act.shape[1], 6), np.float32)
    for h in range(act.shape[0]):
        for w in range(act.shape[1]):
            start_h = h - 2
            end_h = h + 2
            start_w = w - 2
            end_w = w + 2
            if start_h > 0 and end_h > 0 and end_h < act.shape[0] and end_w < act.shape[1]:  # noqa
                responses = act[start_h: end_h, start_w: end_w].reshape(1, -1)
                if responses.shape[1] == 2048:
                    if normalize:
                        responses = (responses - means) / stds
                    predictions = inv(clf.T @ clf) @ clf.T @ responses.T
                    predictions = predictions.squeeze()
                    sel_def[h, w] = predictions[3]
                    argmaxes[h, w] = np.argmax(predictions)
                    tcs[h, w] = predictions
    ams = []
    for idx in range(tcs.shape[-1]):
        ams.append(np.unravel_index(tcs[..., idx].ravel().argmax(), sel_def.shape))
    # am = np.unravel_index(sel_def.argmax(), sel_def.shape)
    # print("Argmax h {} w {}".format(am[0], am[1]))
    print(fi, f)
    print(ams)

    f = plt.figure()
    plt.subplot(141)
    plt.imshow(im)
    plt.axis("off")
    plt.subplot(142)
    plt.imshow(argmaxes)
    plt.axis("off")
    plt.title("Argmax population")
    ax = plt.subplot(143)
    plt.imshow(sel_def)
    plt.axis("off")
    colors = ["r", "g", "b", "y", "k", "c"]
    for ami, color in zip(ams, colors):
        circle = plt.Circle((ami[::-1]), 4, color=color, fill=False)
        ax.add_artist(circle)
    # circle1 = plt.Circle((121, 110), 4, color='r', fill=False)
    # circle2 = plt.Circle((66, 84), 4, color='y', fill=False)
    # circle3 = plt.Circle((180, 73), 4, color='g', fill=False)
    # ax.add_artist(circle1)
    # ax.add_artist(circle2)
    # ax.add_artist(circle3)
    plt.title("$90^{\circ} tuning$")
    ax = plt.subplot(144)
    plt.axis("off")
    plt.title("Connectome")
    op = np.load("BSDS_inh_natural_0_0_BSDS_portrait_connectome_tc_0_0_optim.npy")
    op = op.squeeze().mean(-1)
    # a0 = plt.imshow(op, cmap="RdBu", vmax=12, vmin=-12)
    plt.imshow(sel_def)
    plt.imshow(op, alpha=1. - (op == 0).astype(np.float32), cmap="RdBu", vmax=12, vmin=-12)  # noqa
    # cbar1 = f.colorbar(a0, ax=ax)
    plt.show()
    plt.close(f)
    np.save("bsds_tcs", tcs)
    np.save("bsds_sel_def", sel_def)



