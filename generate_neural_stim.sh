rm -rf plaid_surround
rm -rf plaid_no_surround
rm -rf orientation_tilt
rm -rf orientation_probe
rm -rf orientation_probe_no_surround
rm -rf orientation_probe_no_surround_theta_0
rm -rf contrast_modulated
rm -rf contrast_modulated_no_surround

python tb_stim_wrapper.py 1 1 1 plaid_surround
python tb_stim_wrapper.py 1 1 1 plaid_no_surround
python tb_stim_wrapper.py 1 1 1 orientation_tilt
python tb_stim_wrapper.py 1 1 1 orientation_probe
python tb_stim_wrapper.py 1 1 1 orientation_probe_no_surround
python tb_stim_wrapper_fixed.py 1 1 1 orientation_probe_no_surround_theta_0
python bc_stim_wrapper.py 1 1 1 contrast_modulated
python bc_stim_wrapper.py 1 1 1 contrast_modulated_no_surround

scp -r plaid_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r plaid_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_tilt dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe_no_surround_theta_0 dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r contrast_modulated dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r contrast_modulated_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims

