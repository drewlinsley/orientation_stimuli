import time
import sys
import numpy as np
import os
import contrast_tb_stim  as tb_stim  # import tilt_illusion

class Args:
    def __init__(self):
        self.save_images = True
        self.save_metadata = True

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
args.batch_id = int(sys.argv[2])
dataset_root = str(sys.argv[4])
contrast = float(sys.argv[5])
# total_images = int(sys.argv[3])
# args.n_images = total_images/num_machines

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
# args.lambda_range = [22, 23]  # [30, 90]
args.lambda_range = [16, 17]  # [30, 90]

# args.theta1_range = [22.5, 67.5]  # H/TD
# args.theta2_range = [22.5, 67.5]  # H/TD
args.theta1_range = [-90, 90]  # H/TD
args.theta2_range = [-90, 90]  # H/TD
# args.TB_stim = True
dual_centers = [180]
control_stim = False
surround = True
surround_control = False
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
else:
    raise NotImplementedError(dataset_root)

# dual_centers = [0]  # Only shows orientation
# dual_centers = [180]  # T&B-style stimuli
# control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)

contrasts = [0.06, 0.12, 0.25, 0.5, 0.75, 0.99]
contrasts = [0.75, 0.99]

# for contrast in contrasts:
args.contrast_range = [contrast]  # [contrast, contrast + 0.01]
dataset_subpath = 'train'
it_dataset_root = "optim_contrast_{}_{}".format(contrast, dataset_root)

################################# train
# args.dataset_path = os.path.join(it_dataset_root, dataset_subpath)
# tb_stim.from_wrapper(args, train=True, dual_centers=dual_centers, control_stim=control_stim, surround=surround)

################################# test
dataset_subpath = 'test'
# args.r1_range = [args.r1_range[0]/2, args.r1_range[1]/2]

# ALIGNED WITH TB2015
args.r1_range = [args.r1_range[0]/4, args.r1_range[1]/4]  # Remember 500 -> 224 resize for the model
args.dataset_path = os.path.join(it_dataset_root, dataset_subpath)
tb_stim.from_wrapper(args, train=False, dual_centers=dual_centers, control_stim=control_stim, surround=surround, surround_control=surround_control)

