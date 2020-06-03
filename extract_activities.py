import numpy as np
import re
import os
import sys
from glob import glob
from sklearn import linear_model
import pandas as pd
from sklearn.utils.extmath import safe_sparse_dot                                                    
from joblib import dump, load
MASTER_MODEL_NAME = "linear_model.joblib"
import scipy as sp
from scipy import stats
# DEFAULT_FDOMAIN = (0., 1.)#(-sp.pi, sp.pi)
# DEFAULT_SCALE_CIRCULAR = 1.0/sp.sqrt(5.)
# NOISE_LOC = 0.0
# NOISE_SCALE = 0.0
DEFAULT_FDOMAIN = 0
DEFAULT_SCALE_CIRCULAR = 0


def get_population_circular(stimulus, npoints=180, fdomain=DEFAULT_FDOMAIN,
    scale=DEFAULT_SCALE_CIRCULAR):
    """"""

    unit2ang = lambda a: (a - fdomain[0]) \
        /(fdomain[1]-fdomain[0]) * 2 * sp.pi - sp.pi

    # Create 'npoints' regularly-spaced tuning curves in the desired range 'fdomain'
    z, (h, w) = sp.linspace(fdomain[0], fdomain[1], npoints), stimulus.shape
    population = sp.zeros((npoints, h, w))

    z = unit2ang(z)
    s = unit2ang(stimulus)

    for preferred, sl in zip(z, population):
        kappa = 1.0 / scale ** 2
        sl[:] = stats.vonmises.pdf(s, loc=preferred, kappa=kappa)

    return population


def orientation_diff(array1, array2):
    concat = np.concatenate((np.expand_dims(array1, axis=1),
                             np.expand_dims(array2, axis=1)), axis=1)
    diffs = np.concatenate((np.expand_dims(concat[:,0] - concat[:,1], axis=1),
                            np.expand_dims(concat[:,0] - concat[:,1] - 180, axis=1),
                            np.expand_dims(concat[:,0] - concat[:,1] + 180, axis=1)), axis=1)
    diffs_argmin = np.argmin(np.abs(diffs), axis=1)
    return [idiff[argmin] for idiff, argmin in zip(diffs, diffs_argmin)]


def cluster_points(xs, ys, stepsize):
    xss = list(xs)
    sort_args = np.array(xss).argsort()
    xss.sort()
    ys_sorted = np.array(ys)[sort_args]

    x_accumulator = []
    y_mu = []
    y_25 = []
    y_75 = []
    x_perbin = []
    y_perbin = []
    icut = -90 + stepsize

    for ix, iy in zip(xss, ys_sorted):
        if ix < icut:
            x_perbin.append(ix)
            y_perbin.append(iy)
        else:
            if len(y_perbin) > 0:
                x_accumulator.append(icut - stepsize / 2)
                y_mu.append(np.median(y_perbin))
                y_25.append(np.percentile(y_perbin, 25))
                y_75.append(np.percentile(y_perbin, 75))
            icut += stepsize
            x_perbin = []
            y_perbin = []
    return x_accumulator, y_mu, y_25, y_75


def collapse_points(cs_diff, out_gt_diff):
    cs_diff_collapsed =[]
    out_gt_diff_collapsed = []
    for ix, iy in zip(cs_diff, out_gt_diff):
        if ix < -10:
            cs_diff_collapsed.append(-ix)
            out_gt_diff_collapsed.append(-iy)
        else:
            cs_diff_collapsed.append(ix)
            out_gt_diff_collapsed.append(iy)
    return cs_diff_collapsed, out_gt_diff_collapsed


def screen(r1, lambda1, theta, r1min=None, r1max=None, lambda1min=None, lambda1max=None, thetamin=None, thetamax=None):
    if np.array(r1).size > 1:
        cond = np.ones_like(r1).astype(np.bool)
    else:
        cond = True
    if r1min is not None:
        cond = cond * (r1 > r1min)
    if r1max is not None:
        cond = cond * (r1 < r1max)
    if lambda1min is not None:
        cond = cond * (lambda1 > lambda1min)
    if lambda1max is not None:
       cond = cond * (lambda1 < lambda1max)
    if thetamin is not None:
        cond = cond * ((theta > thetamin) | (theta > thetamin+180))
    if thetamax is not None:
        cond = cond * (theta < thetamax)
    return cond


# im_sub_path, im_fn, iimg,
# r1, theta1, lambda1, shift1,
# r2, theta2, lambda2, shift2, dual_center):


def main():
    import matplotlib.pyplot as plt

    file_name = sys.argv[1]
    meta_dim = int(sys.argv[2])
    analysis = sys.argv[3]
    output_name = sys.argv[4]
    if len(sys.argv) >= 6:
        feature_idx = sys.argv[5]
    paths = glob(os.path.join(file_name, "*.npz"))
    meta = glob(os.path.join(file_name, "*.npy"))[0]
    paths.sort(key=os.path.getmtime)
    path = paths[-1]
    out_data = np.load(path, allow_pickle=True, encoding="latin1")

    out_data_arr = out_data['test_dict']
    meta_arr = np.reshape(
        np.load(meta, allow_pickle=True), [-1, meta_dim])

    # thetas = meta_arr[:, 4].astype(np.float32)
    # plaids = meta_arr[:, -1].astype(np.float32)
    # unique_thetas = np.unique(thetas)

    image_paths = np.asarray(
        [x['image_paths'][0].split(os.path.sep)[-1] for x in out_data_arr])
    target_image_paths = meta_arr[:, 1]
    assert np.all(image_paths == target_image_paths)

    responses = []
    for d in out_data_arr:
        responses.append(d['ephys'].squeeze(0))
    responses = np.asarray(responses)

    if analysis == "feature_select":
        # # Train a linear model to get preferred orientation partitions

        # First get image orientations
        thetas = np.asarray([float(x.split("_")[-1].split(".")[0]) for x in image_paths])

        # Second quantize those into 10 degree bins
        bin_degrees = 1.
        num_bins = int(len(thetas) / bin_degrees)
        bins = pd.qcut(thetas, num_bins, labels=False)
        # indicator_matrix = np.eye(num_bins)[bins]
        # indicator_matrix = get_population_circular(indicator_matrix)
        # indicator = np.arange(num_bins)
        # indicator_matrix = np.asarray([np.roll(indicator, x) for x in range(num_bins)])
        nChannels = 20;
        exponent = 7;
        orientations = np.arange(180);
        prefOrientation = np.arange(0, 180, 180/nChannels)
        basis_vectors = np.zeros((180,nChannels))
        for iChannel in range(nChannels): 
            basis =  np.cos(2*np.pi*(orientations - prefOrientation[iChannel])/180);
            basis[basis<0] = 0;
            basis = basis ** exponent;
            basis_vectors[:,iChannel] = basis;
        indicator_matrix = basis_vectors
        model = linear_model.LogisticRegression(random_state=0, penalty='none', multi_class='multinomial')
        # model = linear_model.LinearRegression()
        means = responses.mean()
        stds = responses.std()
        # responses = (responses - means) / stds
        import ipdb;ipdb.set_trace()
        clf = model.fit(responses, indicator_matrix)
        dump(clf, MASTER_MODEL_NAME)

        # Third, plot Y and X to share with lax
        f = plt.figure()
        plt.subplot(131)
        plt.title('Y')
        plt.imshow(indicator_matrix)
        plt.subplot(132)
        plt.title('X')
        plt.imshow(responses)
        plt.subplot(133)
        plt.title('Parameters')
        plt.imshow(clf.coef_.T)
        plt.show()
        plt.close(f)

        # cutoff = responses.std(0)
        # idx = np.argsort(
        #     cutoff)[::-1][:int(responses.shape[1] * 0.05)]
        # idx = responses.std(0) > cutoff
        # np.savez(output_name, idx=idx, responses=responses)
        np.savez(output_name, means=means, stds=stds)
    elif analysis == "activity":
        # filters = np.load(feature_idx)
        # idx = filters['idx']
        # optimal_orientation = np.argmax(
        #     filters['responses'].sum(-1) / filters['responses'].std(-1))
        # # responses = responses[:, idx]
        # responses = responses[optimal_orientation:, idx]
        moments = np.load(feature_idx)
        means = moments['means']
        stds = moments['stds']
        responses = (responses - means) / stds
        clf = load(MASTER_MODEL_NAME)
        responses = safe_sparse_dot(
            responses,
            clf.coef_.T,
            dense_output=True) + clf.intercept_
        np.save(output_name, responses)
    else:
        raise NotImplementedError(analysis)
    print("Finished {}".format(file_name))

if __name__ == '__main__':
    main()

