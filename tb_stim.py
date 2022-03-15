import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.draw import circle
import os
import imageio
from tqdm import tqdm
import pickle
from skimage.transform import rotate

TB_LIST = [-90, -60, -30, 30, 60, 90]
# TB_LIST = [-30]


def create_annuli_mask(r, imsize):
    rr, cc = circle(imsize[0]//2, imsize[1]//2, r)
    rr_cut = rr[(rr <= (imsize[0]-1)) & (rr >= 0) & (cc <= (imsize[1]-1)) & (cc >= 0)]
    cc_cut = cc[(rr <= (imsize[0]-1)) & (rr >= 0) & (cc <= (imsize[1]-1)) & (cc >= 0)]
    obj_strt_img = np.zeros((imsize[0], imsize[1]))
    obj_strt_img[rr_cut, cc_cut] = 1
    return obj_strt_img


def create_sinusoid(imsize, theta, lmda, shift, amplitude=1.):
    radius = (int(imsize[0]/2.0), int(imsize[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]), range(-radius[1], radius[1]))
    omega = [np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)]
    stimuli = amplitude * np.cos(omega[0] * x * 2 * np.pi/lmda + omega[1] * y * 2 * np.pi/lmda + shift * np.pi/180)
    stimuli += 1
    stimuli = stimuli ** 2
    stimuli = stimuli / stimuli.max()
    stimuli *= 2
    stimuli -= 1
    return stimuli


def create_image(
        imsize,
        r1,
        theta1,
        lambda1,
        shift1,
        gilbert_mask=False,
        gilbert_train=False,
        gilbert_offset=False,
        gilbert_repulse=False,
        gilbert_shift=False,
        gilbert_box=False,
        flip_polarity=False,
        r2=None,
        theta2=None,
        lambda2=None,
        shift2=None,
        dual_center=False,
        timo_type=False,
        timo_contrast_div=False,
        timo_surround_contrast_div=False,
        stride=50,
        offset=20,
        surround=True,
        mask_center=False,
        t_surround=False,
        flanker_offset=0,
        middle_mask_size=1.75,  # Was 1.5
        roll_surround=False,
        image_rotate=False,
        both_flankers=False,
        kapadia_contrast=[0.2, 0.6],
        surround_control=False):
    if gilbert_mask:
        shift1, shift2 = 350,350
    mask1 = create_annuli_mask(r1, imsize)
    sin1 = create_sinusoid(imsize, theta1, lambda1, shift1)
    if dual_center and shift2 is not None:
        if dual_center == True:
            dual_center = lambda1 + np.random.choice(TB_LIST)
            lambda2 = lambda1
            shift2 = shift1
        sindual = create_sinusoid(imsize, dual_center, lambda2, shift2)
        # TODO: Put the random TB_LIST into the meta file
        sin1 = (np.stack((sin1, sindual), -1)).mean(-1)
    if r2 is not None:
        if r2<=r1:
            raise ValueError('r2 should be greater than r1')
        mask2 = create_annuli_mask(r2, imsize)
        sin2 = create_sinusoid(imsize, theta2, lambda2, shift2)
        if not surround:
            mask2 = np.zeros_like(mask2)
        image = sin2*mask2
    else:
        image = np.zeros((imsize[0], imsize[1]))
    image = image*(1-mask1) + sin1*(mask1)
    if r2 is not None:
        # Middle_mask
        middle_mask = create_annuli_mask(r1 * middle_mask_size, imsize)
        if not surround_control:
            middle_mask = middle_mask - mask1
        image[middle_mask > 0] = 0
    if gilbert_mask and r2 is not None:
        # Mask the image
        mask = np.copy(image)
        mask[:, :245] = -1000
        mask[:, 255:] = -1000
        # mask[:, :249] = -1000
        # mask[:, 252:] = -1000
        mask[:235] = -1000
        mask[265:] = -1000
        mask = mask != -1000
        mask = rotate(mask, theta1, order=0, preserve_range=True)
        image = image * mask.astype(np.float32) * (1 - mask2 + mask1)
        image[image > 0] = 1.
        patch_hw = np.where(mask1)
        # patch_hw[0][np.argmin(patch_hw[0])] -= 5
        # patch_hw[0][np.argmax(patch_hw[0])] += 5
        # patch_hw[1][np.argmin(patch_hw[1])] -= 5
        # patch_hw[1][np.argmax(patch_hw[1])] += 5

        patch = image[patch_hw[0].min(): patch_hw[0].max(), patch_hw[1].min(): patch_hw[1].max()]
        og_patch = patch.copy()
        oppo_patch = rotate(patch, -theta2, order=1, preserve_range=True)
        patch = rotate(patch, theta2, order=1, preserve_range=True)
        mask_hw = np.where(mask2)
        if gilbert_train:
            image = np.zeros_like(image)
            image[patch_hw[0].min(): patch_hw[0].max(), patch_hw[1].min(): patch_hw[1].max()] = patch
        else:
            # Now take out the center and move it to the surround

            # image[mask_hw[0].min(): mask_hw[0].min() + patch.shape[0], patch_hw[1].min(): patch_hw[1].max()] = patch
            if gilbert_offset:
                # image[mask_hw[0].max() - patch.shape[0]: mask_hw[0].max(), patch_hw[1].min() - patch.shape[1] // 2: patch_hw[1].min() + patch.shape[1] // 2] = patch
                # image[mask_hw[0].min(): mask_hw[0].min() + patch.shape[0], patch_hw[1].max() - patch.shape[1] // 2: patch_hw[1].max() + patch.shape[1] // 2] = patch
                l = np.percentile(patch_hw[1], 90).astype(int)
                image[mask_hw[0].max() - patch.shape[0]: mask_hw[0].max(), l - patch.shape[1] : l] = patch
                l = np.percentile(patch_hw[1], 10).astype(int)
                image[mask_hw[0].min(): mask_hw[0].min() + patch.shape[0], l: l + patch.shape[1]] = patch
            elif gilbert_shift == 1:
                l = np.percentile(patch_hw[1], 90).astype(int)
                image[mask_hw[0].max() - patch.shape[0]: mask_hw[0].max(), l - patch.shape[1] : l] = patch
                image[mask_hw[0].min(): mask_hw[0].min() + patch.shape[0], l - patch.shape[1]: l] = oppo_patch
            elif gilbert_shift == -1:
                l = np.percentile(patch_hw[1], 10).astype(int)
                image[mask_hw[0].max() - patch.shape[0]: mask_hw[0].max(), l: l + patch.shape[1]] = patch

                line_loc = np.where(image > 1e-4)
                mid = np.median(line_loc[1]).astype(int)
                lhs_l = int(mid - 1.2 * patch.shape[1])
                lhs_r = int(mid + 0.2 * patch.shape[1]) 
                # lhs_l = np.where(mask2)[1].min()
                # lhs_r = image.shape[1] - lhs_l  # np.where(mask2)[1].max() + patch_hw[1].max()
                image[patch_hw[0].min(): patch_hw[0].min() + patch.shape[0], lhs_l: lhs_l + patch.shape[1]] = patch
                image[patch_hw[0].min(): patch_hw[0].min() + patch.shape[0], lhs_r: lhs_r + patch.shape[1]] = patch
            else:
                # Center the patch
                lhs = patch_hw[1].min()  #  + (patch_hw[1].max() - patch_hw[1].min()) // 20

                if t_surround:
                    import pdb;pdb.set_trace()
                    patch = np.maximum(np.roll(patch, -len(patch) // 4, axis=0), rotate(patch, theta2, order=1, preserve_range=True))
                if roll_surround:
                    patch = np.roll(patch, -len(patch) // roll_surround, axis=0)
                if flanker_offset:
                    image[mask_hw[0].max() - patch.shape[0] + flanker_offset: mask_hw[0].max() + flanker_offset, lhs: lhs + patch.shape[1]] = patch  # [flanker_offset:-flanker_offset, flanker_offset:-flanker_offset]
                else:
                    image[mask_hw[0].max() - patch.shape[0]: mask_hw[0].max(), lhs: lhs + patch.shape[1]] = patch
                if both_flankers:
                    # Line below adds a flanker to the top -- not needed for Kapadia 95. Needed for Kinoshita 2008.
                    if flanker_offset:
                        image[mask_hw[0].min() - flanker_offset: mask_hw[0].min() + patch.shape[0] - flanker_offset, lhs: lhs + patch.shape[1]] = patch
                    else:
                        image[mask_hw[0].min(): mask_hw[0].min() + patch.shape[0], lhs: lhs + patch.shape[1]] = patch

            if timo_contrast_div:
                assert timo_contrast_div <= 1 and timo_contrast_div > 0, "gilbert_div must be 1<= and >0"
                image[patch_hw[0].min(): patch_hw[0].max(), patch_hw[1].min(): patch_hw[1].max()] *= timo_contrast_div
            if timo_surround_contrast_div:
                assert timo_surround_contrast_div <= 1 and timo_surround_contrast_div > 0, "gilbert_surround_div must be 1<= and >0"
                # vmin = image.min()
                # vmax = image.max()
                # from matplotlib import pyplot as plt;plt.imshow(image, vmin=vmin, vmax=vmax);plt.show()
                image[:patch_hw[0].min(), :] *= timo_surround_contrast_div
                image[:, patch_hw[1].max():] *= timo_surround_contrast_div
                image[patch_hw[0].max():, :] *= timo_surround_contrast_div
                image[:, :patch_hw[1].min()] *= timo_surround_contrast_div
                # from matplotlib import pyplot as plt;plt.imshow(image, vmin=vmin, vmax=vmax);plt.show()

            if timo_type == "diagonal":
                assert gilbert_offset
                num_paddles = 3
                l = np.percentile(patch_hw[1], 90).astype(int)
                top_flanker = [
                    mask_hw[0].max() - patch.shape[0],
                    mask_hw[0].max(),
                    l - patch.shape[1],
                    l]
                l = np.percentile(patch_hw[1], 10).astype(int)
                bottom_flanker = [
                    mask_hw[0].min(),
                    mask_hw[0].min() + patch.shape[0],
                    l,
                    l + patch.shape[1]]
                top_flanker = np.asarray(top_flanker)
                bottom_flanker = np.asarray(bottom_flanker)
                stride = 50
                for paddle in range(num_paddles):
                    top_flanker[:2] += stride
                    top_flanker[2:] -= stride // 3
                    bottom_flanker[:2] -= stride
                    bottom_flanker[2:] += stride // 3
                    image[top_flanker[0]: top_flanker[1], top_flanker[2]: top_flanker[3]] += patch
                    image[bottom_flanker[0]: bottom_flanker[1], bottom_flanker[2]: bottom_flanker[3]] += patch
                # plt.imshow(image)
                # plt.show()
            elif timo_type == "straight":
                assert not gilbert_offset
                num_paddles = 2  # Should be + from straight
                l = patch_hw[1].min()  #  + (patch_hw[1].max() - patch_hw[1].min()) // 20
                top_flanker = [
                    mask_hw[0].max() - patch.shape[0],
                    mask_hw[0].max(),
                    l,
                    l + patch.shape[1]]
                bottom_flanker = [
                    mask_hw[0].min(),
                    mask_hw[0].min() + patch.shape[0],
                    l,
                    l + patch.shape[1]]
                top_flanker = np.asarray(top_flanker)
                bottom_flanker = np.asarray(bottom_flanker)

                # Kill previous ecrfs
                image[:, :patch_hw[1].min()] = 0
                image[:, patch_hw[1].max():] = 0
                image[:patch_hw[0].min():, :] = 0
                image[patch_hw[0].max():, :] = 0

                # Add rotated patch
                image[patch_hw[0].min():patch_hw[0].max(), patch_hw[1].min(): patch_hw[1].max()] = patch * timo_contrast_div
                for paddle in range(num_paddles):
                    sign = ((paddle % 2) * 2) - 1
                    it_patch = og_patch
                    image[top_flanker[0]: top_flanker[1], top_flanker[2] - offset: top_flanker[3] - offset] = it_patch
                    it_patch = og_patch
                    image[bottom_flanker[0]: bottom_flanker[1], bottom_flanker[2] + offset: bottom_flanker[3] + offset] = it_patch
                    top_flanker[:2] += stride
                    # top_flanker[2:] -= stride // 3
                    bottom_flanker[:2] -= stride
                    # bottom_flanker[2:] += stride // 3
                # plt.imshow(image)
                # plt.show()
            elif timo_type == "zigzag":
                assert not gilbert_offset
                num_paddles = 2  # Should be + from straight
                l = patch_hw[1].min()  #  + (patch_hw[1].max() - patch_hw[1].min()) // 20
                top_flanker = [
                    mask_hw[0].max() - patch.shape[0],
                    mask_hw[0].max(),
                    l,
                    l + patch.shape[1]]
                bottom_flanker = [
                    mask_hw[0].min(),
                    mask_hw[0].min() + patch.shape[0],
                    l,
                    l + patch.shape[1]]
                top_flanker = np.asarray(top_flanker)
                bottom_flanker = np.asarray(bottom_flanker)

                # Kill previous ecrfs
                image[:, :patch_hw[1].min()] = 0
                image[:, patch_hw[1].max():] = 0
                image[:patch_hw[0].min():, :] = 0
                image[patch_hw[0].max():, :] = 0

                # Add rotated patch
                image[patch_hw[0].min():patch_hw[0].max(), patch_hw[1].min(): patch_hw[1].max()] = patch * timo_contrast_div 
                for paddle in range(num_paddles):
                    sign = ((paddle % 2) * 2) - 1
                    it_patch = rotate(patch, -theta2 + (15 * sign), order=1, preserve_range=True)
                    image[top_flanker[0]: top_flanker[1], top_flanker[2] - offset: top_flanker[3] - offset] = it_patch
                    it_patch = rotate(patch, -theta2 + (15 * sign), order=1, preserve_range=True)
                    image[bottom_flanker[0]: bottom_flanker[1], bottom_flanker[2] + offset: bottom_flanker[3] + offset] = it_patch
                    top_flanker[:2] += stride
                    # top_flanker[2:] -= stride // 3
                    bottom_flanker[:2] -= stride
                    # bottom_flanker[2:] += stride // 3
                # plt.imshow(image)
                # plt.show()
            elif timo_type == "spiral":
                assert gilbert_offset
                num_paddles = 4
                l = np.percentile(patch_hw[1], 90).astype(int)
                top_flanker = [
                    mask_hw[0].max() - patch.shape[0],
                    mask_hw[0].max(),
                    l - patch.shape[1],
                    l]
                l = np.percentile(patch_hw[1], 10).astype(int)
                bottom_flanker = [
                    mask_hw[0].min(),
                    mask_hw[0].min() + patch.shape[0],
                    l,
                    l + patch.shape[1]]
                top_flanker = np.asarray(top_flanker)
                bottom_flanker = np.asarray(bottom_flanker)
                stride = 40
                for paddle in range(num_paddles):
                    image[top_flanker[0]: top_flanker[1], top_flanker[2]: top_flanker[3]] = patch
                    image[bottom_flanker[0]: bottom_flanker[1], bottom_flanker[2]: bottom_flanker[3]] = patch
                    if paddle > 1:
                        top_flanker[:2] -= stride
                        top_flanker[2:] -= stride
                        bottom_flanker[:2] += stride
                        bottom_flanker[2:] += stride
                    else:
                        top_flanker[:2] += stride
                        top_flanker[2:] -= stride
                        bottom_flanker[:2] -= stride
                        bottom_flanker[2:] += stride

                plt.imshow(image)
                plt.show()
            elif timo_type:
                raise NotImplementedError(timo_type)
        # Now set all small values to 0 and big values to 1
        image = (image > 0.5).astype(image.dtype)

        if mask_center:
            image = image * (1 - mask)

        if image_rotate:
            image = (rotate(image, image_rotate, order=1, preserve_range=True) > .0).astype(np.float32)
            mask = (rotate(mask, image_rotate, order=1, preserve_range=True) > .0).astype(np.float32)

        # Adjust contrasts for kapadia 95
        if gilbert_train:
            mask = (rotate(mask, theta2, order=1, preserve_range=True) > 0).astype(mask.dtype)
        # mask = rotate(mask, theta2, order=0, preserve_range=True) 
        kapadia_center_contrast, kapadia_surround_contrast = kapadia_contrast
        image[mask == 1] *= kapadia_center_contrast  # 0.2
        image[mask != 1] *= kapadia_surround_contrast  # 0.6

    else:
        # Still offer the opportunity to change contrasts
        if timo_contrast_div < 1. and timo_type:
            assert timo_contrast_div <= 1 and timo_contrast_div > 0, "gilbert_div must be 1<= and >0"
            patch_hw = np.where(mask1)
            h_min, h_max = patch_hw[0].min() - 10, patch_hw[0].max() + 10
            w_min, w_max = patch_hw[1].min() - 10, patch_hw[1].max() + 10
            image[patch_hw[0].min(): patch_hw[0].max(), patch_hw[1].min(): patch_hw[1].max()] *= timo_contrast_div
    image += 1
    image *= 127.5
    if gilbert_box:
        # Draw a box around the r1-sized RF
        # boxr = int(r1 + image.shape[0] * 0.02)
        boxr = int(r1 - image.shape[0] * 0.0)
        center = image.shape[0] // 2
        imm = image.max()
        lw = 1
        image[center - boxr : center - boxr + lw + lw, center - boxr: center + boxr] = imm
        image[center + boxr - lw - lw: center + boxr, center - boxr: center + boxr] = imm
        image[center - boxr: center + boxr, center - boxr: center - boxr + lw + lw] = imm
        image[center - boxr: center + boxr, center + boxr - lw - lw: center + boxr] = imm
    return image


def accumulate_meta(array,
                    im_sub_path, im_fn, iimg,
                    r1, theta1, lambda1, shift1,
                    r2, theta2, lambda2, shift2, dual_center):
    # NEW VERSION
    array += [im_sub_path, im_fn, iimg,
              r1, theta1, lambda1, shift1,
              r2, theta2, lambda2, shift2, dual_center]
    return array


def save_metadata(metadata, contour_path, batch_id):
    # Converts metadata (list of lists) into an nparray, and then saves
    metadata_path = os.path.join(contour_path, 'metadata')
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    metadata_fn = str(batch_id) + '.npy'
    # np.save(os.path.join(metadata_path,metadata_fn), metadata, fix_imports=True)
    with open(os.path.join(metadata_path, metadata_fn), "wb") as f:
        pickle.dump(metadata, f, protocol=2)


def from_wrapper(args, train=True, dual_centers=[90], control_stim=False, surround=True, include_null=False, surround_control=False, gilbert_mask=False, gilbert_train=False, gilbert_offset=False, gilbert_repulse=False, gilbert_shift=False, flip_polarity=False, gilbert_box=False, timo_type=False, timo_contrast_div=False, timo_surround_contrast_div=False, stride=50, offset=20, mask_center=False, t_surround=False, roll_surround=False, kapadia_contrast=[[0.2, 0.6]], image_rotate=False, both_flankers=False):
    import time
    import scipy

    if (args.save_images):
        im_sub_path = os.path.join('imgs', str(args.batch_id))
        if not os.path.exists(os.path.join(args.dataset_path, im_sub_path)):
            os.makedirs(os.path.join(args.dataset_path, im_sub_path))
    if args.save_metadata:
        metadata = []
        # CHECK IF METADATA FILE ALREADY EXISTS
        metadata_path = os.path.join(args.dataset_path, 'metadata')
        if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
        metadata_fn = str(args.batch_id) + '.npy'
        metadata_full = os.path.join(metadata_path, metadata_fn)
        if os.path.exists(metadata_full):
            print('Metadata file already exists.')
            return

    r1_range = np.arange(args.r1_range[0], args.r1_range[1])
    lambda_range = np.arange(args.lambda_range[0], args.lambda_range[1])
    theta1_range = np.arange(*args.theta1_range)  # args.theta1_range[0], args.theta1_range[1])
    theta2_range = np.arange(*args.theta2_range)  # args.theta2_range[0], args.theta2_range[1])
    flanker_offset_range = np.arange(*args.flanker_offset_range)
    if image_rotate:
        image_rotate_range = np.arange(*image_rotate)
    else:
        image_rotate_range = [False]
    combos = [[i, j, k, l, m, n] for i in r1_range 
                 for j in lambda_range 
                 for k in theta1_range
                 for l in flanker_offset_range
                 for m in kapadia_contrast
                 for n in image_rotate_range]
    for iimg, combo in tqdm(enumerate(combos), total=len(combos), desc="Building dataset"):
        t = time.time()

        # Correct label names
        if len(kapadia_contrast) == 1:
            iimg = 180 - iimg
        im_fn = "sample_%s.png" % (iimg)

        ##### SAMPLE IMAGE PARAMETERS
        r1, lmda, theta1, flanker_offset, kapadia_contrast, image_rotate = combo
        theta2 = theta1
        shift1 = 180  # 180
        if train:
            r2=None
            theta2=None
            shift2=None
        else:
            r2 = r1 * 4
            # r2 = r1 * 2
        theta2 = theta1
        shift2 = shift1
        for dual_center in dual_centers:
            # dual_center += theta1
            if control_stim:
                theta1 = dual_center
                dual_center = False
            img = create_image(
                args.image_size,
                r1, theta1, lmda, shift1,
                r2=r2, theta2=theta2, lambda2=lmda, shift2=shift2, dual_center=dual_center, surround=surround, surround_control=surround_control, gilbert_mask=gilbert_mask, gilbert_train=gilbert_train, gilbert_offset=gilbert_offset, gilbert_repulse=gilbert_repulse, gilbert_shift=gilbert_shift, flip_polarity=flip_polarity, gilbert_box=gilbert_box, timo_type=timo_type, timo_contrast_div=timo_contrast_div, timo_surround_contrast_div=timo_surround_contrast_div, stride=stride, offset=offset, flanker_offset=flanker_offset, mask_center=mask_center,t_surround=t_surround, roll_surround=roll_surround, kapadia_contrast=kapadia_contrast, image_rotate=image_rotate, both_flankers=both_flankers)

            if (args.save_images):
                imageio.imwrite(os.path.join(args.dataset_path, im_sub_path, im_fn), img)
            if (args.save_metadata):
                metadata = accumulate_meta(
                    metadata,
                    im_sub_path, im_fn, iimg,
                    r1, theta1, lmda, shift1,
                    r2, theta2, lmda, shift2, dual_center=dual_center)
            elapsed = time.time() - t
            # print('PER IMAGE : ', str(elapsed))
    if (args.save_metadata):
        #print(metadata)
        matadata_nparray = np.array(metadata)
        save_metadata(matadata_nparray, args.dataset_path, args.batch_id)

    return


if __name__ == "__main__":
    # generate test sample image
    params = {'imsize': [500, 500],
              'r1': 100,  # 80-240
              'theta1': 60,  # 30-90
              'lambda1': 30,
              'shift1': 0,
              'r2': 200,
              'theta2': 60,
              'dual_center': 180,
              'lambda2': 30,
              'shift2': 0}
    im1 = create_image(**params)
    plt.imshow(im1, cmap='Greys_r');plt.show()
