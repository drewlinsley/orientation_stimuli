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
# args.lambda_range = [16, 17]  # [30, 90]

# args.theta1_range = [22.5, 67.5]  # H/TD
# args.theta2_range = [22.5, 67.5]  # H/TD
args.theta1_range = [-90, 90]  # H/TD
args.theta2_range = [-90, 90]  # H/TD

args.flanker_offset_range = [0, 1]

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
mask_center = False
t_surround = False
timo_type = False
roll_surround = False
image_rotate = False
both_flankers = False
timo_contrast_div = 1.
timo_surround_contrast_div = 1.

# kapadia_contrast = [[0.3, 0.4]]  # [[0.2, 0.6]]
kapadia_contrast = [[0.1, 0.2]]  # [[0.4, 0.4]]  # [[0.2, 0.6]]

stride = 50
offset = 20

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
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = True
    flip_polarity = True
elif dataset_root == "gilbert_angelluci_flanker_offsets":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    image_rotate = [-90, -89]
    args.theta1_range = [0, 1]
    args.theta2_range = [0, 1]
    args.flanker_offset_range = [-18, -17]
elif dataset_root == "gilbert_angelluci_flanker_kinoshita":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    image_rotate = [-90, -89]
    both_flankers = True
    args.theta1_range = [0, 1]
    args.theta2_range = [0, 1]
    # args.flanker_offset_range = [-30, 22, 2]
    args.flanker_offset_range = [-18, -17]
elif dataset_root == "gilbert_angelluci_flanker_contrast_offsets":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    kapadia_contrast = [
        [0.1, 0.],
        [0.1, 0.2],
        [0.1, 0.4],
        [0.1, 0.6],
    ]
    image_rotate = [-90, 90, 90]
    args.theta1_range = [0, 1]
    args.theta2_range = [0, 1]
    # args.theta1_range = [-90, 90, 30]
    # args.theta2_range = [-90, 90, 30]
    args.flanker_offset_range = [-30, 22, 2]
elif dataset_root == "gilbert_angelluci_flanker_rotations":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    image_rotate = [-90, -89]
    args.theta1_range = [-90, 90]
    args.theta2_range = [-90, 90]
    # args.flanker_offset_range = [-30, -29]  # The argmax from gilbert_angelluci_flanker_offsets
    args.flanker_offset_range = [-18, -17]
elif dataset_root == "gilbert_angelluci_flanker_only":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    mask_center = True
    image_rotate = [-90, -89]
    args.theta1_range = [0, 1]
    args.theta2_range = [0, 1]
    # args.flanker_offset_range = [-30, -29]  # The argmax from gilbert_angelluci_flanker_offsets
    args.flanker_offset_range = [-18, -17]
elif dataset_root == "gilbert_angelluci_horizontal_flanker_only":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    mask_center = True
    roll_surround = 4
    image_rotate = [-90, -89]
    args.theta1_range = [-90, -89]
    args.theta2_range = [-90, -89]
    # args.flanker_offset_range = [-30, -29]  # The argmax from gilbert_angelluci_flanker_offsets
    args.flanker_offset_range = [-18, -17]
elif dataset_root == "gilbert_angelluci_t_flanker_only":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    mask_center = True
    image_rotate = [-90, -89]
    args.theta1_range = [-90, -89]
    args.theta2_range = [-90, -89]
    # args.flanker_offset_range = [-30, -29]  # The argmax from gilbert_angelluci_flanker_offsets
    args.flanker_offset_range = [-18, -17]
    t_surround = True
elif dataset_root == "gilbert_angelluci_t_flanker":
    # args.lambda_range = [120, 121]
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_box = False
    flip_polarity = True
    image_rotate = [-90, -89]
    args.theta1_range = [-90, -89]
    args.theta2_range = [-90, -89]
    # args.flanker_offset_range = [-30, -29]  # The argmax from gilbert_angelluci_flanker_offsets
    args.flanker_offset_range = [-18, -17]
    t_surround = True
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

elif dataset_root == "timo_straight_high_contrast_stride_40":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 0.2
    stride = 40

elif dataset_root == "timo_straight_high_contrast_stride_60":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 0.2
    stride = 60

elif dataset_root == "timo_straight_high_contrast_stride_70":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 0.2
    stride = 70

elif dataset_root == "timo_straight_high_contrast_stride_80":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 0.2
    stride = 80

elif dataset_root == "timo_straight_low_contrast_stride_40":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    stride = 40

elif dataset_root == "timo_straight_low_contrast_stride_60":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    stride = 60

elif dataset_root == "timo_straight_low_contrast_stride_70":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    stride = 70

elif dataset_root == "timo_straight_low_contrast_stride_80":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    stride = 80


elif dataset_root == "timo_straight_low_contrast_offset_0":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    offset = 0

elif dataset_root == "timo_straight_low_contrast_offset_10":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    offset = 10

elif dataset_root == "timo_straight_low_contrast_offset_30":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    offset = 30

elif dataset_root == "timo_straight_low_contrast_offset_40":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = 1.
    offset = 40

elif dataset_root == "timo_straight_high_contrast_training":
    dual_centers = [0]  # Only shows orientation
    surround = False
    timo_contrast_div = .5
    timo_type = "straight"

elif dataset_root == "timo_straight_high_contrast_offset_0":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = .25
    timo_surround_contrast_div = 0.5
    offset = 0

elif dataset_root == "timo_straight_high_contrast_offset_10":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = .25
    timo_surround_contrast_div = 0.5
    offset = 10

elif dataset_root == "timo_straight_high_contrast_offset_20":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = .25
    timo_surround_contrast_div = 0.5
    offset = 20

elif dataset_root == "timo_straight_high_contrast_offset_30":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = .25
    timo_surround_contrast_div = 0.5
    offset = 30

elif dataset_root == "timo_straight_high_contrast_offset_40":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    surround = True
    gilbert_mask = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "straight"
    timo_contrast_div = .25
    timo_surround_contrast_div = 0.5
    offset = 40

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
    timo_contrast_div = 1.

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
    gilbert_train = False
    gilbert_offset = False
    # flip_polarity = True
    timo_type = "zigzag"
    timo_contrast_div = 1.

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
    timo_contrast_div = 1.

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
    timo_contrast_div = 1.

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
tb_stim.from_wrapper(args, train=False, dual_centers=dual_centers, control_stim=control_stim, gilbert_mask=gilbert_mask, gilbert_train=gilbert_train, surround=surround, surround_control=surround_control, gilbert_offset=gilbert_offset, gilbert_repulse=gilbert_repulse, gilbert_shift=gilbert_shift, flip_polarity=flip_polarity, gilbert_box=gilbert_box, timo_type=timo_type, timo_contrast_div=timo_contrast_div, timo_surround_contrast_div=timo_surround_contrast_div, stride=stride, offset=offset, mask_center=mask_center, t_surround=t_surround, roll_surround=roll_surround, kapadia_contrast=kapadia_contrast, image_rotate=image_rotate)

