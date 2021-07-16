import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage.draw import circle
import os
import imageio
from tqdm import tqdm
import pickle


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

def create_image(imsize,
                 r1, theta1, lambda1, shift1,
                 r2=None, theta2=None, lambda2=None, shift2=None, dual_center=False, surround=True, surround_control=False, contrast=None):
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
        middle_mask = create_annuli_mask(r1 * 1.5, imsize)
        if not surround_control:
            middle_mask = middle_mask - mask1
        image[middle_mask > 0] = 0
    image = image * contrast
    image += 1
    image *= 127.5
    return image / 255.

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

def from_wrapper(args, train=True, dual_centers=[90], control_stim=False, surround=True, include_null=False, surround_control=False):  # -90, -60, -30, 0, 60]):
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
    theta1_range = np.arange(args.theta1_range[0], args.theta1_range[1])
    theta2_range = np.arange(args.theta2_range[0], args.theta2_range[1])
    # contrast_range = np.arange(args.contrast_range[0], args.contrast_range[1])
    combos = [[i, j, k, l]
                 for i in r1_range 
                 for j in lambda_range
                 for k in args.contrast_range
                 for l in theta1_range]
    for iimg, combo in tqdm(enumerate(combos), total=len(combos), desc="Building dataset"):
        t = time.time()

        # Correct label names
        iimg = 180 - iimg
        im_fn = "sample_%s.png" % (iimg)

        ##### SAMPLE IMAGE PARAMETERS
        r1, lmda, contrast, theta1 = combo
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
                r1, theta1, lmda, shift1, contrast=contrast,
                r2=r2, theta2=theta2, lambda2=lmda, shift2=shift2, dual_center=dual_center, surround=surround, surround_control=surround_control)

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

