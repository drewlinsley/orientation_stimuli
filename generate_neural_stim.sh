rm -rf plaid_surround
rm -rf plaid_no_surround
rm -rf orientation_tilt
rm -rf orientation_probe
rm -rf contrast_modulated

python tb_stim_wrapper.py 1 1 1 plaid_surround
python tb_stim_wrapper.py 1 1 1 plaid_no_surround
python tb_stim_wrapper.py 1 1 1 orientation_tilt
python tb_stim_wrapper.py 1 1 1 orientation_probe
python bc_stim_wrapper.py 1 1 1 contrast_modulated

scp -r plaid_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r plaid_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_tilt dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r contrast_modulated dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims

