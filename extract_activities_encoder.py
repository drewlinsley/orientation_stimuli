import numpy as np
import os
import sys
from glob import glob
from sklearn import linear_model, cross_decomposition
import pandas as pd
from sklearn.utils.extmath import safe_sparse_dot
from sklearn import decomposition, metrics
from tqdm import tqdm
from joblib import dump, load
from scipy import stats
# import scipy as sp
# from scipy import stats
# DEFAULT_FDOMAIN = (0., 1.)#(-sp.pi, sp.pi)
# DEFAULT_SCALE_CIRCULAR = 1.0/sp.sqrt(5.)
# NOISE_LOC = 0.0
# NOISE_SCALE = 0.0
DEFAULT_FDOMAIN = 0
DEFAULT_SCALE_CIRCULAR = 0
MASTER_MODEL_NAME = "linear_model.joblib"
MASTER_PREPROC_NAME = "preproc.joblib"


def main(basis_function="cos", model_type="gauss", debug=True, decompose=False, normalize=False, cross_validate=False):
    import matplotlib.pyplot as plt

    file_name = sys.argv[1]
    meta_dim = int(sys.argv[2])
    analysis = sys.argv[3]
    output_name = sys.argv[4]
    if len(sys.argv) >= 6:
        feature_idx = sys.argv[5]
    save_images = False
    if len(sys.argv) >= 7:
        save_images = True
    paths = glob(os.path.join(file_name, "*.npz"))
    meta = glob(os.path.join(file_name, "*.npy"))[0]
    paths.sort(key=os.path.getmtime)
    path = paths[-1]
    out_data = np.load(path, allow_pickle=True, encoding="latin1")

    out_data_arr = out_data['test_dict']
    meta_arr = np.reshape(
        np.load(meta, allow_pickle=True, encoding='latin1'), [-1, meta_dim])

    # thetas = meta_arr[:, 4].astype(np.float32)
    # plaids = meta_arr[:, -1].astype(np.float32)
    # unique_thetas = np.unique(thetas)

    image_paths = np.asarray(
        [str(x['image_paths'][0]).split(os.path.sep)[-1].strip("'") for x in out_data_arr])  # noqa
    target_image_paths = meta_arr[:len(image_paths), 1]
    # assert np.all(image_paths == target_image_paths)
    image_paths = image_paths.astype(target_image_paths.dtype)
    if np.all(image_paths != target_image_paths):
        print("Model image_paths are different than meta image paths")

    responses = []
    images = []
    for d in out_data_arr:
        responses.append(d['prepre_ephys'].squeeze(0))
        # responses.append(d['pre_ephys'].squeeze(0))
        if save_images:
            images.append(d['images'].squeeze())
        # responses.append(d['logits'].squeeze(0))
        # plt.imshow(d['images'].squeeze()[..., 0], cmap='Greys_r');plt.show()
        # responses.append(d['ephys'].squeeze(0))
        # responses.append(d['pool_act_5max'].squeeze())
        # responses.append(d['pool_act_10max'].squeeze())
        # responses.append(d['pool_act_5mean'].squeeze())
        # responses.append(d['pool_act_10mean'].squeeze())

        # responses.append(d['fpool_act_5max'].squeeze())
        # responses.append(d['fpool_act_10max'].squeeze())
        # responses.append(d['fpool_act_5mean'].squeeze())
        # responses.append(d['fpool_act_10mean'].squeeze())

    if save_images:
        output_image_name = "images_{}".format(output_name)
        np.save(output_image_name, images)
        print("Saved images to {}".format(output_image_name))

    responses = np.asarray(responses)
    if np.any(responses < 0):
        print("Rectifying responses.")
    responses = np.maximum(np.asarray(responses), 0.)

    # # Align meta with the activities (slow)
    # aligner = []
    # for meta_row in meta_arr:
    #     for i, p in enumerate(image_paths):
    #         if meta_row[1] == p:
    #             aligner.append(i)
    #             break
    # aligner = np.asarray(aligner)
    # import ipdb;ipdb.set_trace()
    # responses = responses[aligner]

    if analysis == "feature_select":
        cutoff = (responses.mean(0) / (responses.std(0) + 1e-8))
        idx = np.argsort(cutoff)[::-1][:int(responses.shape[1] * (1.))]
        # cutoff = np.abs(responses.mean(0))
        # idx = np.argsort(cutoff)[::-1][:int(responses.shape[1] * (1. / 3.))]

        # idx = np.argsort(cutoff)[::-1][:int(responses.shape[1] * 0.33)]
        # idx = np.where(np.abs(cutoff) > 2)[0]
        print("Keeping {} features".format(len(idx)))
        responses = responses[:, idx]
        # np.savez(output_name, idx=idx, responses=responses)

        # # Train a linear model to get preferred orientation partitions

        # First get image orientations
        # thetas = np.unique(np.asarray(
        #     [
        #         float(x.split("_")[-1].split(".")[0])
        #         for x in image_paths]))
        meta_arr = meta_arr[:len(responses)]
        # meta_arr = meta_arr[:500]
        # responses = responses[:500]
        thetas = np.unique(meta_arr[:, 4])

        # Second quantize those into 10 degree bins
        bin_degrees = 1
        num_bins = int(len(thetas) / bin_degrees)
        bins = pd.qcut(thetas, num_bins, labels=False)
        if basis_function == "step":
            indicator_matrix_plot = np.eye(num_bins)[bins]
            indicator_matrix = bins
            if model_type != "logistic":
                indicator_matrix = indicator_matrix_plot
            indicator_matrix = np.roll(
                indicator_matrix, bin_degrees // 2, axis=0)
            # indicator_matrix_plot = indicator_matrix
        elif basis_function == "cos":
            nChannels = num_bins
            exponent = 1
            orientations = np.arange(180)
            prefOrientation = np.arange(0, 180, 180 / nChannels)
            indicator_matrix = np.zeros((180, nChannels))
            for iChannel in range(nChannels):
                basis = np.cos(2 * np.pi * (
                    orientations - prefOrientation[iChannel]) / 180)
                # basis[basis < 0] = 0
                basis = basis ** exponent
                indicator_matrix[:, iChannel] = basis
            indicator_matrix_plot = indicator_matrix
        elif basis_function == "gauss":
            indicator_matrix = np.zeros((180, 180))
            prefOrientation = np.arange(0, 180, 180)
            gaussian = stats.norm(loc=90, scale=5).pdf(np.arange(180))  # noqa
            for i in range(180):
                indicator_matrix[:, i] = np.roll(gaussian, i - 90)
            indicator_matrix_plot = indicator_matrix
        else:
            raise NotImplementedError(basis_function)
        if model_type == "logistic":
            model = linear_model.LogisticRegression(random_state=0, multi_class='ovr', max_iter=1000)
        elif model_type == "linear":
            model = linear_model.LinearRegression(fit_intercept=True, normalize=False)
        elif model_type == "lasso":
            # model = linear_model.MultiTaskLassoCV(cv=2, normalize=True, n_jobs=2, fit_intercept=True, verbose=True, max_iter=100000, random_state=0)
            model = linear_model.Lasso(positive=True, alpha=0.0001, fit_intercept=False, max_iter=1000000, random_state=0)
            # model = linear_model.MultiTaskLasso(alpha=0.1, fit_intercept=False, max_iter=100000, random_state=0)
        elif model_type == "ridge":
            model = linear_model.Ridge(normalize=False, alpha=0.10, fit_intercept=False)
        elif model_type == "pls":
            model = cross_decomposition.PLSRegression(n_components=12, scale=False, tol=1e-12, max_iter=1000)
        elif model_type == "argmax":
            model = None
        else:
            raise NotImplementedError(model_type)

        # Remove 180 degrees
        meta_col = meta_arr[:, 4].astype(int)
        if meta_col.max() <= 90:
            meta_col = meta_col + 90
        else:
            pass
            # meta_col = meta_col - 90
        # meta_col[meta_col == 179] = 0
        print("max-meta: {}, min-meta: {}".format(meta_col.max(), meta_col.min()))

        meta_col = meta_col % 180
        print(meta_col.max(), meta_col.min())
        # meta_col[meta_col == 180] = 0
        # meta_col[meta_col == 181] = 1
        # indicator_matrix = indicator_matrix[:, :-1]
        indicator_matrix = indicator_matrix[meta_col]

        # Make the target nonnegative
        # indicator_matrix = np.maximum(indicator_matrix, 0)
        indicator_matrix = indicator_matrix - indicator_matrix.min()
        indicator_matrix = indicator_matrix / indicator_matrix.max()
        indicator_matrix_plot = indicator_matrix

        if decompose:
            preproc = decomposition.NMF(n_components=10, random_state=0, verbose=False, alpha=0.5, max_iter=10000)
            # preproc = decomposition.FastICA(n_components=8, random_state=0, whiten=True, max_iter=10000)
            # preproc = decomposition.PCA(n_components=128, random_state=0, whiten=False)
            original_responses = np.copy(responses)
            preproc.fit(responses)
            responses = preproc.transform(responses)
            dump(preproc, MASTER_PREPROC_NAME)

        means = responses.mean()
        stds = responses.std() + 1e-12
        if normalize:
            responses = (responses - means) / stds
            # responses = (responses - means)  #  / stds
        # indicator_matrix = (indicator_matrix - indicator_matrix.mean(0))
        # indicator_matrix = (indicator_matrix - indicator_matrix.mean(0)) / (indicator_matrix.std(0) + 1e-12)
        if "both" in file_name:
            diff = responses.shape[0] - indicator_matrix.shape[0]
            indicator_matrix = np.concatenate((indicator_matrix, indicator_matrix[0][None].repeat(diff, 0)), 0)  # noqa
            indicator_matrix_plot = indicator_matrix

        if cross_validate and model_type == "pls":
            rsquared = []
            for nc in tqdm(range(responses.shape[-1]), desc="Cross validating PLS", total=responses.shape[-1]):   # noqa
                model = cross_decomposition.PLSRegression(
                    n_components=nc + 1,
                    scale=True,
                    tol=1e-12,
                    max_iter=10000)
                clf = model.fit(responses, indicator_matrix)
                preds = clf.predict(responses)
                rsquared.append(
                    metrics.r2_score(y_true=indicator_matrix, y_pred=preds))
            rsquared = np.asarray(rsquared)
            components = (
                np.diff(rsquared) > 0.00001).sum()  # noqa Threshold at 99.9% of cumvar
            print("Keeping {} components".format(components))
            model = cross_decomposition.PLSRegression(
                n_components=components + 1,
                scale=True,
                tol=1e-12,
                max_iter=10000)
        elif cross_validate and model_type == "lasso":
            fits = []
            for nc in tqdm(range(2, original_responses.shape[-1] // 4), desc="Cross validating NMF", total=original_responses.shape[-1] // 4):   # noqa
                preproc = decomposition.NMF(n_components=nc, random_state=0, verbose=False, alpha=0.01, max_iter=10000)  # noqa
                preproc.fit(original_responses)
                fits.append(preproc.reconstruction_err_)
            plt.plot(fits)
            plt.show()
            os._exit(1)
        elif cross_validate:
            raise RuntimeError("cross_validate is True but model is not PLS.")
        if model is None:
            # Selected argmax
            clf = np.argmax(responses, 1)
        else:
            # clf = model.fit(indicator_matrix, responses)  # noqa Encoder models like serences
            clf = np.matmul(np.linalg.pinv(indicator_matrix), responses)
            # clf = model.fit(responses, indicator_matrix)  #
        dump(clf, MASTER_MODEL_NAME)
        np.savez(output_name, idx=idx, means=means, stds=stds)

        # Third, plot Y and X to share with lax
        if debug:
            f = plt.figure()
            plt.subplot(131)
            plt.title('X')
            plt.imshow(indicator_matrix_plot)
            plt.subplot(133)
            plt.title('Y')
            plt.imshow(responses)
            plt.subplot(132)
            plt.title('Parameters')
            plt.imshow(clf)
            plt.show()
            plt.close(f)

    elif analysis == "activity":
        moments = np.load(feature_idx)
        idx = moments['idx']
        # optimal_orientation = np.argmax(
        #     filters['responses'].sum(-1) / filters['responses'].std(-1))
        # responses = responses[optimal_orientation:, idx]
        means = moments['means']
        stds = moments['stds']
        responses = responses[:, idx]
        # responses = np.maximum(responses, 0) / responses.max()
        # assert (responses < 0).sum() == 0
        clf = load(MASTER_MODEL_NAME)
        if decompose:
            preproc = load(MASTER_PREPROC_NAME)
            responses = preproc.transform(responses)
        if normalize:
            responses = (responses - means) / stds
            # responses = (responses - means)  # / stds
        if model_type == "pls":
            responses = clf.predict(responses)
        elif model_type == "argmax":
            # "electrophys"
            responses = responses[clf]
        else:
            params = safe_sparse_dot(
                clf,
                clf.T,
                dense_output=True)
            inv_params = np.linalg.pinv(params)
            responses = safe_sparse_dot(
                responses,
                clf.T,
                dense_output=True)  # + clf.intercept_
            predictions = safe_sparse_dot(
                responses,
                inv_params,
                dense_output=True)
        # responses = stds * (responses + means)
        # assert (responses < 0).sum() == 0
        np.save(output_name, predictions)
    else:
        raise NotImplementedError(analysis)
    print("Finished {}".format(file_name))


if __name__ == '__main__':
    main()
