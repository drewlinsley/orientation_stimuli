import numpy as np


d = np.load("model_outputs/gammanet_full_contrast_modulated_no_surround_outputs_data.npy")
np.save("model_outputs/gammanet_full_contrast_modulated_no_surround_outputs_data_flipped", d[::-1])
