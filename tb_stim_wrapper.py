import time
import sys
import numpy as np
import os
import tb_stim  # import tilt_illusion

class Args:
    def __init__(self):
        self.save_images = True
        self.save_metadata = True

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
args.batch_id = int(sys.argv[2])
# total_images = int(sys.argv[3])
# args.n_images = total_images/num_machines

if len(sys.argv)==4:
    print('Using default path...')
    dataset_root = 'images'  # '/Users/junkyungkim/Desktop/tilt_illusion'
elif len(sys.argv)==5:
    print('Using custom save path...')
    dataset_root = str(sys.argv[4])
else:
    raise ValueError('wrong number of args')

## Parameters
args.image_size = [500, 500]
# args.r1_range = [110, 111]  # [100, 120]
# args.r1_range = [119, 120]  # [100, 120]
# args.r1_range = [150, 151]  # [100, 120]

# args.r1_range = [80, 81]  # [100, 120]
# args.lambda_range = [30, 31]  # [30, 90]
# args.lambda_range = [40, 41]  # [30, 90]
# args.lambda_range = [44, 45]  # [30, 90]
# args.lambda_range = [40, 41]  # [30, 90]
# args.lambda_range = [60, 61]  # [30, 90]


# ALIGNED WITH TB2015
args.r1_range = [31 * 3, (31 * 3) + 1]  # [100, 120]
# args.lambda_range = [15, 16]  # [30, 90]
args.lambda_range = [22, 23]  # [30, 90]

# args.theta1_range = [22.5, 67.5]  # H/TD
# args.theta2_range = [22.5, 67.5]  # H/TD
args.theta1_range = [-90, 90]  # H/TD
args.theta2_range = [-90, 90]  # H/TD
# args.TB_stim = True
dual_centers = [180]
control_stim = False
surround = True
surround_control = False
gilbert_mask = False
gilbert_train = False
gilbert_offset = False
gilbert_repulse = False
gilbert_shift = False
gilbert_box = False
flip_polarity = False
timo_type = False
timo_contrast_div = False

if dataset_root == "plaid_surround":
    dual_centers = [180]  # T&B-style stimuli
elif dataset_root == "plaid_no_surround":
    dual_centers = [180]  # T&B-style stimuli
    surround = False
elif dataset_root == "orientation_tilt":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
elif dataset_root == "orientation_probe":
    dual_centers = [0]  # Only shows orientation
elif dataset_root == "orientation_probe_no_surround":
    dual_centers = [0]  # Only shows orientation
    surround = False
elif dataset_root == "surround_control":
    dual_centers = [180]
    surround_control = True
elif dataset_root == "gilbert_angelluci_offset":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = True
elif dataset_root == "gilbert_angelluci_train_offset":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
    gilbert_offset = True
elif dataset_root == "gilbert_angelluci_right":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_shift = 1
elif dataset_root == "gilbert_angelluci_left":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_shift = -1
elif dataset_root == "gilbert_angelluci":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
elif dataset_root == "gilbert_angelluci_box":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = True
elif dataset_root == "gilbert_angelluci_repulse":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_repulse = True
elif dataset_root == "gilbert_angelluci_train":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
elif dataset_root == "gilbert_angelluci_train_box":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
    gilbert_box = True
elif dataset_root == "flip_gilbert_angelluci_offset":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = True
    flip_polarity = True

elif dataset_root == "flip_gilbert_angelluci_train_offset":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
    gilbert_offset = True
    flip_polarity = True

elif dataset_root == "flip_gilbert_angelluci_right":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_shift = 1
    flip_polarity = True

elif dataset_root == "flip_gilbert_angelluci_left":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_shift = -1
    flip_polarity = True

elif dataset_root == "flip_gilbert_angelluci":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    flip_polarity = True

elif dataset_root == "flip_gilbert_angelluci_repulse":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_repulse = True
    flip_polarity = True

elif dataset_root == "flip_gilbert_angelluci_train":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
    flip_polarity = True

elif dataset_root == "timo_straight_high_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 0.2

elif dataset_root == "timo_straight_high_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 0.2

elif dataset_root == "timo_straight_low_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = False

elif dataset_root == "timo_zigzag_high_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "zigzag"
    timo_contrast_div = 0.2

elif dataset_root == "timo_zigzag_low_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "zigzag"
    timo_contrast_div = False

elif dataset_root == "timo_diagonal_high_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = True
    gilbert_offset = True
    flip_polarity = True
    timo_type = "diagonal"
    timo_contrast_div = 0.2

elif dataset_root == "timo_diagonal_low_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = True
    flip_polarity = True

    timo_type = "diagonal"
    timo_contrast_div = False

elif dataset_root == "timo_spiral_high_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = True
    flip_polarity = True
    timo_type = "spiral"
    timo_contrast_div = 0.2

elif dataset_root == "timo_spiral_low_contrast":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = True
    flip_polarity = True

    timo_type = "spiral"
    timo_contrast_div = False

else:
    raise NotImplementedError(dataset_root)

# dual_centers = [0]  # Only shows orientation
# dual_centers = [180]  # T&B-style stimuli
# control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)

################################# train
dataset_subpath = 'train'
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
tb_stim.from_wrapper(args, train=True, dual_centers=dual_centers, control_stim=control_stim, surround=surround)

################################# test
dataset_subpath = 'test'
# args.r1_range = [args.r1_range[0]/2, args.r1_range[1]/2]

# ALIGNED WITH TB2015
args.r1_range = [args.r1_range[0]/4, args.r1_range[1]/4]  # Remember 500 -> 224 resize for the model
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
if dataset_root == "gilbert":
    surround = True
tb_stim.from_wrapper(args, train=False, dual_centers=dual_centers, control_stim=control_stim, gilbert_mask=gilbert_mask, gilbert_train=gilbert_train, surround=surround, surround_control=surround_control, gilbert_offset=gilbert_offset, gilbert_repulse=gilbert_repulse, gilbert_shift=gilbert_shift, flip_polarity=flip_polarity, gilbert_box=gilbert_box, timo_type=timo_type, timo_contrast_div=timo_contrast_div)

