import time
import sys
import numpy as np
import os
import bc_stim  # import tilt_illusion

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
args.r1_range = [150, 151]  # [100, 120]
# args.r1_range = [80, 81]  # [100, 120]
# args.lambda_range = [30, 31]  # [30, 90]
args.lambda_range = [60, 61]  # [30, 90]
# args.theta1_range = [22.5, 67.5]  # H/TD
# args.theta2_range = [22.5, 67.5]  # H/TD
# args.theta1_range = [-90, -89]  # H/TD
# args.theta2_range = [-90, -89]  # H/TD
args.theta1_range = [-45, -44]  # H/TD
args.theta2_range = [-45, -44]  # H/TD
control_stim = False

args.contrast_range = [0., 0.06, 0.12, 0.25, 0.5]
dual_centers = [45]
surround, train = True, True
if dataset_root == "contrast_modulated_no_surround":
    surround = False
    train = False
################################# train
dataset_subpath = 'train'
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
bc_stim.from_wrapper(args, train=train, dual_centers=dual_centers, surround=surround)

################################# test
dataset_subpath = 'test'
args.r1_range = [args.r1_range[0]/2, args.r1_range[1]/2]
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
bc_stim.from_wrapper(args, train=False, dual_centers=dual_centers, control_stim=control_stim, surround=surround)

