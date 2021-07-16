"""Train and test inverted encoding models."""
import numpy as np
import os
from glob import glob
from joblib import dump, load
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def main(
        file_name,
        meta_dim,
        channels,
        model_output,
        train_moments,
        train_model=False,
        null_npz="responses/contrast_modulated_no_surround_outputs",
        save_images=False,
        model_file="linear_model.joblib",
        preproc_file="preproc.joblib",
        basis_function="cos",
        model_type="linear",
        pool_type=None,
        debug=False,
        decompose=False,
        normalize=False,
        inv=np.linalg.inv,
        start_idx=1280,
        idx_stride=128,
        population=False,
        include_null=False,
        exp_diff=3,
        feature_select=1.,
        cross_validate=False):
    """Create and test IEM models."""

    # Find and load paths
    paths = glob(os.path.join(file_name, "*.npz"))
    meta = glob(os.path.join(file_name, "*.npy"))[0]
    paths.sort(key=os.path.getmtime)
    path = paths[-1]
    out_data = np.load(path, allow_pickle=True, encoding="latin1")
    out_data_arr = out_data['test_dict']
    meta_arr = np.reshape(
        np.load(meta, allow_pickle=True, encoding='latin1'), [-1, meta_dim])
    image_paths = np.asarray(
        [str(x['image_paths'][0]).split(os.path.sep)[-1].strip("'") for x in out_data_arr])  # noqa
    target_image_paths = meta_arr[:len(image_paths), 1]
    # assert np.all(image_paths == target_image_paths)
    image_paths = image_paths.astype(target_image_paths.dtype)
    if np.all(image_paths != target_image_paths):
        print("Model image_paths are different than meta image paths")

    # Prepare output files
    assert ".npz" in train_moments, "train_moments must have a .npz extension."

    # Store model responses
    extract_key = "fpool_act_5max"  # First out of GN
    if population:
        units = np.arange(2048)  # 4x4 cube
    else:
        units = np.arange(start_idx, start_idx + idx_stride, dtype=int)

    # Find units to mask
    sample = out_data_arr[0][extract_key]
    sample = sample.squeeze().mean(-1)
    mask = sample > sample.mean()
    mask[:10] = False
    mask[-10:] = False
    mask[:, :10] = False
    mask[:, -10:] = False

    # Loop through units to make models
    coords = np.where(mask)
    for i, (h, w) in enumerate(zip(coords[0], coords[1])):
        responses = []
        model_file_split = model_file.split(os.path.sep)
        curr_model_file = os.path.join(model_file_split[0], "model_h{}_w{}_{}".format(h, w, model_file_split[1]))  # noqa
        model_moments_split = train_moments.split(os.path.sep)
        curr_train_moments = os.path.join(model_moments_split[0], "moments_h{}_w{}_{}".format(h, w, model_moments_split[1]))  # noqa

        for d in out_data_arr:
            data = d[extract_key].squeeze(0)[h:h + 4, w:w + 4]
            data = data.reshape(-1)
            responses.append(data[units])  # noqa Pre cos/sin 1x1 conv
        responses = np.asarray(responses)
        D = 128
        if pool_type is None:
            responses = responses.reshape(len(responses), -1)
        elif pool_type == "mean":
            responses = responses.reshape(len(responses), D, -1)
            responses = responses.mean(-1)
        elif pool_type == "max":
            responses = responses.reshape(len(responses), D, -1)
            responses = responses.max(-1)
        else:
            raise NotImplementedError(pool_type)

        # Correct meta: Remove 180 degrees and reverse.
        meta_col = meta_arr[:, 4].astype(int)
        if meta_col.min() == -90:
            meta_col = meta_col + 90
        else:
            pass
        meta_col = meta_col % 180
        print("max-meta: {}, min-meta: {}".format(meta_col.max(), meta_col.min()))  # noqa

        # Begin IEM logic
        if train_model:
            print("Meta len: {}, Response len: {}".format(len(meta_arr), len(responses)))  # noqa
            if len(meta_arr) == len(responses):
                print("# responses == # metas")
            else:
                print("# responses IS DIFFERENT # metas")
                meta_arr = meta_arr[:len(responses)]
            # assert len(meta_arr) == len(responses), "Different number of responses and metas."  # noqa
            if basis_function == "cos":
                exponent = channels - exp_diff  # 1
                orientations = np.arange(180)
                prefOrientation = np.linspace(0, 180, channels + 1)[:-1]
                indicator_matrix = np.zeros((180, channels))
                for iChannel, mu in enumerate(prefOrientation):
                    basis = (np.cos(np.pi * (orientations - mu) / 180))
                    # basis[basis < 0] = 0
                    basis = basis ** exponent
                    indicator_matrix[:, iChannel] = basis
                if np.any(indicator_matrix < 0):
                    print("Warning: negative values in the idealized curves. Applying abs.")  # noqa
                    indicator_matrix = np.abs(indicator_matrix)
                indicator_matrix_plot = indicator_matrix
            else:
                raise NotImplementedError(basis_function)
            if model_type != "linear":
                raise NotImplementedError(model_type)
            indicator_matrix = indicator_matrix[meta_col]

            # Stack the null response to the top
            if include_null:
                null_npzs = glob(os.path.join(null_npz, "*.npz"))
                assert len(
                    null_npzs), "No null files found: {}".format(null_npz)
                null_npzs.sort(key=os.path.getmtime)
                null_npz = null_npzs[-1]
                null_response = np.load(null_npz, allow_pickle=True, encoding="latin1")  # noqa
                null_response = null_response["test_dict"]
                null_response = null_response[0][extract_key].squeeze(0)[units].reshape(1, -1)  # noqa Vector response to 0 contrast
                responses = np.concatenate((null_response, responses), 0)
                indicator_matrix = np.concatenate((np.ones((1, indicator_matrix.shape[1])) * indicator_matrix.min(), indicator_matrix), 0)  # noqa
            indicator_matrix_plot = np.copy(indicator_matrix)  # noqa
            # indicator_matrix_plot = indicator_matrix

            # Feature selection
            cutoff = (responses.mean(0) / (responses.std(0) + 1e-8))
            idx = cutoff > -np.inf  # cutoff.mean()
            print(
                "Keeping {}/{} features".format(idx.sum(), responses.shape[1]))
            responses = responses[:, idx]
            # print("Keeping {} features".format(len(idx)))

            # Compute moments
            means = responses.mean()
            stds = responses.std() + 1e-12
            # means = responses.min()
            # stds = responses.max()
            if normalize:
                responses = (responses - means) / stds

            if decompose:
                preproc = decomposition.NMF(n_components=channels, random_state=0, verbose=False, alpha=0., max_iter=10000)  # noqa
                # preproc = decomposition.FastICA(n_components=channels, random_state=0, whiten=True, max_iter=10000)  # noqa
                # preproc = decomposition.PCA(n_components=channels, random_state=0, whiten=False)  # noqa
                preproc.fit(responses)
                responses = preproc.transform(responses)
                dump(preproc, preproc_file)

            if "both" in file_name:
                raise NotImplementedError
                diff = responses.shape[0] - indicator_matrix.shape[0]
                indicator_matrix = np.concatenate((indicator_matrix, indicator_matrix[0][None].repeat(diff, 0)), 0)  # noqa
                indicator_matrix_plot = indicator_matrix

            # Transpose to match Serences
            print("Matrix rank: {}, matrix shape: {}".format(np.linalg.matrix_rank(indicator_matrix), indicator_matrix.shape[1]))  # noqa
            responses = responses.T  # voxels X trials
            indicator_matrix = indicator_matrix.T  # channels X trials
            clf = responses @ indicator_matrix.T @ inv(indicator_matrix @ indicator_matrix.T)  # noqa
            dump(clf, curr_model_file)
            np.savez(curr_train_moments, idx=idx, means=means, stds=stds)

            # Third, plot Y and X to share with lax
            if debug:
                f = plt.figure()
                plt.subplot(141)
                plt.title('Idealized (transpose)')
                plt.imshow(indicator_matrix_plot.T)
                plt.subplot(144)
                plt.title('Y')
                plt.imshow(responses)
                plt.subplot(142)
                plt.title('Parameters')
                plt.imshow(clf)
                plt.subplot(143)
                plt.title('inverse')
                plt.imshow(inv(indicator_matrix @ indicator_matrix.T))
                plt.show()
                plt.close(f)

        else:
            moments = np.load(curr_train_moments)
            means = moments['means']
            stds = moments['stds']
            idx = moments['idx']
            if "orientation_probe_no_surround_outputs" in model_output or "orientation_probe_outputs" in model_output:  # noqa
                if include_null:
                    null_npzs = glob(os.path.join(null_npz, "*.npz"))
                    null_npzs.sort(key=os.path.getmtime)
                    null_npz = null_npzs[-1]
                    null_response = np.load(null_npz, allow_pickle=True, encoding="latin1")  # noqa
                    null_response = null_response["test_dict"]
                    null_response = null_response[0][extract_key].squeeze(0)[units].reshape(1, -1)  # noqa Vector response to 0 contrast
                    responses = np.concatenate((null_response, responses))
                else:
                    responses = np.concatenate((responses[0].reshape(1, -1), responses))  # noqa
            responses = responses[:, idx]
            clf = load(curr_model_file)
            if normalize:
                responses = (responses - means) / stds
            if decompose:
                preproc = load(preproc_file)
                responses = preproc.transform(responses)
            predictions = inv(clf.T @ clf) @ clf.T @ responses.T
            np.save(model_output, predictions)
        print("Finished {}".format(file_name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--responses",
        type=str,
        dest="file_name",
        default=None,
        help="Name of experiment with model responses.")
    parser.add_argument(
        "--null_npz",
        type=str,
        dest="file_name",
        default="responses/contrast_modulated_no_surround_outputs",
        help="Name of null file with baseline response.")
    parser.add_argument(
        "--model_output",
        type=str,
        dest="model_output",
        default=None,
        help="Name of npy file to save predicted outputs.")
    parser.add_argument(
        "--train_moments",
        type=str,
        dest="train_moments",
        default=None,
        help="Name of npz file to save and load moments from training.")
    parser.add_argument(
        "--meta_dim",
        type=int,
        dest="meta_dim",
        default=None,
        help="Number of dimensions in meta file.")
    parser.add_argument(
        "--exp_diff",
        type=int,
        dest="exp_diff",
        default=3,
        help="Difference between exponent and channels.")
    parser.add_argument(
        "--start_idx",
        type=int,
        dest="start_idx",
        default=1280,
        help="Neuron index to start at.")
    parser.add_argument(
        "--idx_stride",
        type=int,
        dest="idx_stride",
        default=128,
        help="Take neurons up to this stride.")
    parser.add_argument(
        "--train_model",
        action="store_true",
        dest="train_model",
        default=False,
        help="Train a model.")
    parser.add_argument(
        "--population",
        action="store_true",
        dest="population",
        default=False,
        help="Get indices for a population cross at the middle of the stim.")
    parser.add_argument(
        "--save_images",
        action="store_true",
        dest="save_images",
        default=False,
        help="Save images for plotting later.")
    parser.add_argument(
        "--model_file",
        type=str,
        dest="model_file",
        default="linear_model.joblib",
        help="Name of npz file where trained model is stored.")
    parser.add_argument(
        "--channels",
        type=int,
        dest="channels",
        default=6,
        help="Number of idealized orientation channels.")
    main(**vars(parser.parse_args()))
