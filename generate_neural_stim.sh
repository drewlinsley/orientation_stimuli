rm -rf plaid_surround
rm -rf surround_control
rm -rf plaid_no_surround
rm -rf orientation_tilt
rm -rf orientation_probe
rm -rf orientation_probe_no_surround
rm -rf orientation_probe_no_surround_theta_0
rm -rf contrast_modulated
rm -rf contrast_modulated_no_surround
rm -rf phase_modulated
rm -rf optim_contrast_*

python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.06
python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.12
python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.25
python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.50
python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.75

python tb_stim_wrapper.py 1 1 1 plaid_surround
python tb_stim_wrapper.py 1 1 1 surround_control
python tb_stim_wrapper.py 1 1 1 plaid_no_surround
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_train
python tb_stim_wrapper.py 1 1 1 orientation_tilt
python tb_stim_wrapper.py 1 1 1 orientation_probe
python tb_stim_wrapper.py 1 1 1 orientation_probe_no_surround
python tb_stim_wrapper_fixed.py 1 1 1 orientation_probe_no_surround_theta_0
python bc_stim_wrapper.py 1 1 1 contrast_modulated
python bc_stim_wrapper.py 1 1 1 contrast_modulated_no_surround
python phase_stim_wrapper.py 1 1 1 phase_modulated
python phase_stim_wrapper.py 1 1 1 phase_modulated_plaid

scp -r optim_contrast_* dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r plaid_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r surround_control dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r plaid_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r gilbert_angelluci dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r gilbert_angelluci_train dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_tilt dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r orientation_probe_no_surround_theta_0 dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r contrast_modulated dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r contrast_modulated_no_surround dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r phase_modulated dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims
scp -r phase_modulated_plaid dlinsley@transfer.ccv.brown.edu:/users/dlinsley/neural_stims

