import os
import numpy as np
from joblib import load
from matplotlib import pyplot as plt
import seaborn as sns  # noqa
from tqdm import tqdm
from skimage.transform import rotate


def tcs(sel_units, clf, inv_clf, perturb):
    """Covert to tuning curves"""
    tc = inv_clf @ clf.T @ sel_units.T
    tc = tc * perturb   # mdu
    return tc


def invert_tcs(tc, clf, inv_inv):
    """Invert the tuning curves into actvities."""
    # tc_inv = tf.matmul(tc, tf.matmul(clf.T, tf.matmul(clf, tf.matmul(clf.T, inv_inv))), transpose_a=True)  # noqa
    tc_inv = (clf.T @ inv_inv @ clf @ clf.T).T @ tc
    tc_inv = np.maximum(tc_inv, 0.).reshape(-1)  # Rectify
    return tc_inv


def get_tuning(perturbations, thetas, conv2_2, clf, inv_clf, inv_inv, circuits, shuffle_theta=False, disable_tqdm=False):  # noqa
    for perturb in tqdm(perturbations, desc="Perturbation", total=len(perturbations), disable=disable_tqdm):  # noqa
        rhos = []
        for theta in thetas:
            act = conv2_2[theta]
            adj_perturb = perturb / 100

            # Perturb the curves
            tc = tcs(act, clf, inv_clf, perturb=adj_perturb)

            # Invert the curves
            inv_tc = invert_tcs(tc, clf, inv_inv)

            # Store the mean pattern
            mean_tc = inv_tc.reshape(16, 128).mean(0)

            # Load the functional connectome
            if shuffle_theta:
                theta = thetas[np.random.permutation(len(thetas))[0]]
            fc = np.load(os.path.join(circuits.format(perturb, perturb, perturb, theta)))  # noqa
            fc = np.abs(fc.squeeze())  # Make I/E positive
            fc = rotate(fc, theta, preserve_range=True, order=1)
            mask = fc.std(-1) > 0.2
            mask_coordinates = np.where(mask)

            # Correlate the center and every other valid location
            it_rho = np.zeros((fc.shape[0], fc.shape[1]))
            for mh, mw in zip(mask_coordinates[0], mask_coordinates[1]):
                fc_v = fc[mh, mw]
                it_rho[mh, mw] = np.corrcoef(mean_tc, fc_v)[0, 1]
            rhos.append(it_rho)
    rhos = np.asarray(rhos)
    return rhos


# Get middle column ff activity under diff perturbations. Correlate every column w this. Plot mean of rhos  # noqa
circuits = "/Users/drewlinsley/Downloads/circuits/circuits_circuit_exc_{}_full_field_orientation_probe/BSDS_exc_{}_perturb_circuit_contrast_circuit_exc_{}_full_field_{}_optim.npy"  # noqa
f = "/Users/drewlinsley/Documents/tilt_illusion/responses/INSILICO_RES/INSILICO_BSDS_vgg_gratings_simple/INSILICO_BSDS_vgg_gratings_simple_orientation_test/"  # noqa
d = np.load(os.path.join(f, "BSDS_vgg_gratings_simple_orientation_test_2020_12_17_13_45_30_536545.npz"), allow_pickle=True, encoding='latin1')  # noqa
rcond = 1e-8  # 1e-4
perturbations = np.asarray([105, 110, 115, 120, 125, 130, 135, 140])
thetas = np.asarray([1, 31, 61, 91, 121, 151])

# Unzip conv encodings
conv2_2 = []
for stim in d["test_dict"]:
    conv2_2.append(stim["conv2_2"])
conv2_2 = np.asarray(conv2_2).squeeze()

# Load linear models for FF drive
clf = load("/Users/drewlinsley/Documents/tilt_illusion/linear_models/INSILICO_BSDS_vgg_gratings_simple_conv2_2_tb_model.joblib")  # noqa
clf = clf.astype(np.float32)
inv_inv = np.linalg.pinv(clf.dot(clf.T), rcond=rcond)
clf_sq = clf.T @ clf
inv_clf = np.linalg.inv(clf_sq).astype(np.float32)  # Precompute inversion

# True tunings
true_tuning = get_tuning(perturbations, thetas, conv2_2, clf, inv_clf, inv_inv, circuits, disable_tqdm=False)  # noqa
mean_true_tuning = true_tuning.mean(0)

# Permuted tunings
iterations = 200
shuff_tunings = []
for idx in tqdm(range(iterations), desc="Permuting tuning", total=iterations):
    shuff_tuning = get_tuning(perturbations, thetas, conv2_2, clf, inv_clf, inv_inv, circuits, shuffle_theta=True, disable_tqdm=True)  # noqa
    shuff_tunings.append(shuff_tuning)
shuff_tunings = np.asarray(shuff_tunings)
shuff_tunings = shuff_tunings.mean(1)
mean_shuff_tunings = shuff_tunings.mean(0)

# Compute p-value at every location
mask = true_tuning.mean(0) > 0
mask_coordinates = np.where(mask)
p_values = np.zeros((true_tuning.shape[1], true_tuning.shape[2]), dtype=np.float32)  # noqa
for mh, mw in zip(mask_coordinates[0], mask_coordinates[1]):
    p_values[mh, mw] = (
        (shuff_tunings[:, mh, mw] > mean_true_tuning[mh, mw]).astype(np.float32).sum() + 1) / (iterations + 1.)  # noqa

plt.subplot(141)
plt.imshow(np.abs(mean_true_tuning) / true_tuning.std(0), cmap="flare", vmin=0, vmax=3)  # noqa
plt.subplot(142)
plt.imshow(mean_true_tuning, cmap="flare")
# plt.colorbar()
plt.subplot(143)
plt.imshow(mean_shuff_tunings, cmap="flare")
plt.subplot(144)
plt.imshow(p_values, cmap="flare")
plt.show()


np.savez("circuit_tuning_results", p_values=p_values, shuff_tunings=shuff_tunings, true_tuning=true_tuning, mask=mask)  # noqa
