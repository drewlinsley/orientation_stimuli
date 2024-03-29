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

# ALIGNED WITH TB2015
args.r1_range = [31 * 3, (31 * 3) + 1]  # [100, 120]
args.lambda_range = [22, 23]  # [30, 90]

# args.theta1_range = [22.5, 67.5]  # H/TD
# args.theta2_range = [22.5, 67.5]  # H/TD
# args.theta1_range = [-90, -89]  # H/TD
# args.theta2_range = [-90, -89]  # H/TD
args.theta1_range = [-1, 0]  # H/TD
control_stim = False

args.contrast_range = [0., 0.06, 0.12, 0.25, 0.5]
surround, train = False, False
################################# train
dataset_subpath = 'train'
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
bc_stim.from_wrapper(args, train=train, dual_centers=dual_centers, surround=surround)

################################# test
dataset_subpath = 'test'
args.r1_range = [args.r1_range[0]/4, args.r1_range[1]/4]  # Remember 500 -> 224 resize for the model
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
bc_stim.from_wrapper(args, train=False, control_stim=control_stim)

