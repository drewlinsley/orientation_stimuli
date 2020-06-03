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
args.r1_range = [110, 111]  # [100, 120]
args.lambda_range = [45, 46]  # [30, 90]
# args.theta1_range = [22.5, 67.5]  # H/TD
# args.theta2_range = [22.5, 67.5]  # H/TD
args.theta1_range = [-90, 90]  # H/TD
args.theta2_range = [-90, 90]  # H/TD
# args.TB_stim = True
dual_centers = [180]
control_stim = False
surround = True
if dataset_root == "plaid_surround":
    dual_centers = [180]  # T&B-style stimuli
elif dataset_root == "plaid_no_surround":
    dual_centers = [180]  # T&B-style stimuli
    surround = False
elif dataset_root == "orientation_tilt":
    control_stim = True  # Produce tilt-illusion-style stim (Fig. 2 of T&B)
elif dataset_root == "orientation_probe":
    dual_centers = [0]  # Only shows orientation
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
args.r1_range = [args.r1_range[0]/2, args.r1_range[1]/2]
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
tb_stim.from_wrapper(args, train=False, dual_centers=dual_centers, control_stim=control_stim, surround=surround)

