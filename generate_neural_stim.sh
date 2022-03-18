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
rm -rf *gilbert_angelluci*
rm -rf timo*

DESTINATION=../refactor_gammanet

# python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.06
# python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.12
# python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.25
# python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.50
# python contrast_tb_stim_wrapper.py 1 1 1 orientation_probe 0.75

python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_train
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_offsets
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_rotations
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_contrast_offsets
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_only
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_t_flanker_only
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_t_flanker
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_horizontal_flanker_only
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_kinoshita

# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_offset
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_left
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_right
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_repulse
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_train_offset
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_train
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_offsets
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_rotations
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_contrast_offsets
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_only
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_t_flanker_only
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_t_flanker
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_horizontal_flanker_only
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_kinoshita

# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_train_box
# python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_box

# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci_offset
# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci
# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci_left
# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci_right
# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci_repulse
# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci_train_offset
# python tb_stim_wrapper.py 1 1 1 flip_gilbert_angelluci_train

# python tb_stim_wrapper.py 1 1 1 timo_straight_high_contrast_stride_80

# python tb_stim_wrapper.py 1 1 1 timo_straight_high_contrast
# python tb_stim_wrapper.py 1 1 1 timo_straight_low_contrast
# python tb_stim_wrapper.py 1 1 1 timo_zigzag_high_contrast
# python tb_stim_wrapper.py 1 1 1 timo_zigzag_low_contrast

# rm -rf orientation_tilt orientation_probe orientation_probe_no_surround plaid_surround plaid_no_surround surround_control
python tb_stim_wrapper.py 1 1 1 orientation_tilt
python tb_stim_wrapper.py 1 1 1 orientation_probe
python tb_stim_wrapper.py 1 1 1 orientation_probe_no_surround
python tb_stim_wrapper.py 1 1 1 plaid_surround
python tb_stim_wrapper.py 1 1 1 plaid_no_surround
python tb_stim_wrapper.py 1 1 1 surround_control

# python tb_stim_wrapper_fixed.py 1 1 1 orientation_probe_no_surround_theta_0
python bc_stim_wrapper.py 1 1 1 contrast_modulated
python bc_stim_wrapper.py 1 1 1 contrast_modulated_no_surround
# python phase_stim_wrapper.py 1 1 1 phase_modulated
# python phase_stim_wrapper.py 1 1 1 phase_modulated_plaid
# python control_contrast_stim_wrapper.py 1 1 1 control_contrast_modulated_no_surround

# Prepare the kapadia dataset
rm -rf kapadia_experiment
cp -rf gilbert_angelluci_train kapadia_experiment
rm kapadia_experiment/test/imgs/1/*
# cp gilbert_angelluci_train/test/imgs/1/sample_1.png kapadia_experiment/test/imgs/1/sample_180.png  # center only
cp gilbert_angelluci_flanker_contrast_offsets/test/imgs/1/sample_0.png kapadia_experiment/test/imgs/1/sample_180.png
cp gilbert_angelluci_flanker_only/test/imgs/1/sample_180.png kapadia_experiment/test/imgs/1/sample_179.png  # flanker only
# cp gilbert_angelluci_flanker_offsets/test/imgs/1/sample_175.png kapadia_experiment/test/imgs/1/sample_178.png
cp gilbert_angelluci_flanker_contrast_offsets/test/imgs/1/sample_14.png kapadia_experiment/test/imgs/1/sample_178.png  # center and flanker
cp gilbert_angelluci_t_flanker/test/imgs/1/sample_180.png kapadia_experiment/test/imgs/1/sample_177.png  # center and T flanker
# cp gilbert_angelluci_t_flanker_only/test/imgs/1/sample_180.png kapadia_experiment/test/imgs/1/sample_176.png  # T flanker only
cp gilbert_angelluci_horizontal_flanker_only/test/imgs/1/sample_180.png kapadia_experiment/test/imgs/1/sample_176.png  # T flanker only
python prepare_kapadia_fig11_data.py

# Fix the gilbert_angelluci_flanker_kinoshita dataset (add a center only stim)
rm gilbert_angelluci_flanker_kinoshita/test/imgs/1/sample_180.png
cp gilbert_angelluci_flanker_contrast_offsets/test/imgs/1/sample_0.png gilbert_angelluci_flanker_kinoshita/test/imgs/1/sample_0.png
python prepare_kinoshita_fig11_data.py

# Move data
cp -rf orientation_tilt ${DESTINATION}
cp -rf orientation_probe ${DESTINATION}
cp -rf orientation_probe_no_surround ${DESTINATION}
cp -rf plaid_surround ${DESTINATION}
cp -rf plaid_no_surround ${DESTINATION}
cp -rf surround_control ${DESTINATION}
cp -rf contrast_modulated* ${DESTINATION}
cp -rf *gilbert* ${DESTINATION}
cp -rf kapadia_experiment ${DESTINATION}

