import os
import numpy as np
from matplotlib import pyplot as plt
from joblib import load
from skimage.transform import rotate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance

def get_curve(activity, moments, means, stds, clf):
    # Normalize activities
    sel_units = (activity - means) / stds

    # Map responses
    inv_clf = np.linalg.inv(clf.T.dot(clf))
    inv_matmul = inv_clf.dot(clf.T)
    sel_units = inv_matmul.dot(sel_units.T)
    return sel_units


version = "exc"  # "inh"
optim_folder = "exc_2.0"  # "inh_0.001"
use_full_field = False
all_args, all_diffs, all_peaks = [], [], []
thetas = [1, 31, 61, 91, 121, 151]
for theta in thetas:
    try:
        # Target
        data = np.load("responses/orientation_probe_outputs/BSDS_vgg_gratings_simple_orientation_test_2020_10_21_11_45_08_472412.npz", allow_pickle=True, encoding="latin1")
        data = data["test_dict"]
        data = np.asarray([x["fpool_act_5max"] for x in data]).squeeze(1)
        clf = load(os.path.join("linear_models", "INSILICO_BSDS_vgg_gratings_simple_tb_model.joblib"))
        moments = np.load(os.path.join("linear_moments", "INSILICO_BSDS_vgg_gratings_simple_tb_feature_matrix.npz"))
        means = moments["means"]
        stds = moments["stds"]
        h, w = data.shape[1], data.shape[2]
        hh, hw = h // 2, w // 2
        activities = data[theta, hh - 2: hh + 2, hw - 2: hw + 2]
        target = get_curve(activities.reshape(1, -1), moments, means, stds, clf)

        # Others
        data = data.squeeze()[theta]
        mask = (data ** 2).mean(-1) > 1.
        mask[:10] = False
        mask[-10:] = False
        mask[:, :10] = False
        mask[:, -10:] = False
        for h in range(4):
            for w in range(4):
                mask[hh + h, hw + w] = False
        he, wi = np.where(mask)
        args = np.zeros((mask.shape[0], mask.shape[1]))
        diffs = np.zeros_like(args)
        peaks = np.zeros_like(args)
        for h, w in zip(he, wi):
            activities = data[h:h + 4, w: w + 4]
            activities = activities.reshape(1, -1)
            try:
                clf = load(os.path.join("linear_models", "model_h{}_w{}_tb_model.joblib".format(h, w))).astype(np.float32)
                moments = np.load(os.path.join("linear_moments", "moments_h{}_w{}_tb_feature_matrix.npz".format(h, w)), allow_pickle=True, encoding="latin1")
                means = moments["means"]
                stds = moments["stds"]
                tc = get_curve(activities, moments, means, stds, clf)
                args[h, w] = np.abs(np.argmax(tc) - np.argmax(target))
                diffs[h, w] = np.mean((tc.squeeze() * target) / (np.linalg.norm(tc) * np.linalg.norm(target)))
                peaks[h, w] = distance.correlation(tc.squeeze(), target)

                # peaks[h, w] = np.abs(tc.squeeze()[np.round(theta / 30).astype(int)] - target[np.round(theta / 30).astype(int)])
                print("Found {}, {}".format(h, w))
            except:
                print("Couldnt find {}, {} -- try p8?".format(h, w))
        all_args.append(rotate(args, -theta, preserve_range=True, order=0))
        all_diffs.append(rotate(diffs, -theta, preserve_range=True, order=0))
        all_peaks.append(rotate(peaks, -theta, preserve_range=True, order=0))
    except:
        print("Failed on theta={}".format(theta))

f, axs = plt.subplots(1, 3, figsize=(16, 4))
im0 = axs[0].imshow(np.stack(all_args).mean(0), cmap="RdBu")
axs[0].set_xticks([])
axs[0].set_yticks([])
divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im0, cax=cax, orientation='vertical')

im1 = axs[1].imshow(np.stack(all_diffs).mean(0)[25:-25, 25:-25], cmap="RdBu")
axs[1].set_xticks([])
axs[1].set_yticks([])
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im1, cax=cax, orientation='vertical')

im2 = axs[2].imshow(np.stack(all_peaks).mean(0), cmap="RdBu")
axs[2].set_xticks([])
axs[2].set_yticks([])
divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
f.colorbar(im2, cax=cax, orientation='vertical')

plt.show()
