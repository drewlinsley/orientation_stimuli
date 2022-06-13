DESTINATION=../refactor_gammanet
rm -rf ../refactor_gammanet/INSILICO_data_all_models/INSILICO_data_all_models_gammanet_bsds_model_1376000
rm -rf orientation_probe_no_surround
rm -rf orientation_tilt
rm -rf orientation_probe
rm -rf plaid_no_surround
rm -rf surround_search
rm -rf surround_control
rm -rf plaid_surround
rm -rf contrast_modulated
rm -rf contrast_modulated_no_surround
rm -rf *gilbert_angelluci*
rm -rf kapadia_experiment

# TB
python tb_stim_wrapper.py 1 1 1 orientation_probe
python tb_stim_wrapper.py 1 1 1 orientation_tilt
python tb_stim_wrapper.py 1 1 1 plaid_surround
python tb_stim_wrapper.py 1 1 1 plaid_no_surround
python tb_stim_wrapper.py 1 1 1 orientation_probe_no_surround
python tb_stim_wrapper.py 1 1 1 surround_search
python tb_stim_wrapper.py 1 1 1 surround_control

# BC
python bc_stim_wrapper.py 1 1 1 contrast_modulated_no_surround
python bc_stim_wrapper.py 1 1 1 contrast_modulated

# Kapadia
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_train
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_offsets
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_rotations
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_contrast_offsets
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_only
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_t_flanker_only
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_t_flanker
python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_horizontal_flanker_only

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

python tb_stim_wrapper.py 1 1 1 gilbert_angelluci_flanker_kinoshita
# Fix the gilbert_angelluci_flanker_kinoshita dataset (add a center only stim)
rm gilbert_angelluci_flanker_kinoshita/test/imgs/1/sample_180.png
cp gilbert_angelluci_flanker_contrast_offsets/test/imgs/1/sample_0.png gilbert_angelluci_flanker_kinoshita/test/imgs/1/sample_0.png
python prepare_kinoshita_fig11_data.py

# Remove old images
rm -rf ${DESTINATION}/orientation_probe*
rm -rf ${DESTINATION}/orientation_tilt
rm -rf ${DESTINATION}/plaid_*surround
rm -rf ${DESTINATION}/surround_search
rm -rf ${DESTINATION}/surround_control
rm -rf ${DESTINATION}/contrast_modulated
rm -rf ${DESTINATION}/contrast_modulated_no_surround
rm -rf ${DESTINATION}/*gilbert*
rm -rf ${DESTINATION}/kapadia_experiment

# Copy over new images
cp -rf orientation_probe* ${DESTINATION}
cp -rf orientation_tilt ${DESTINATION}
cp -rf plaid_*surround ${DESTINATION}
cp -rf surround_search ${DESTINATION}
cp -rf contrast_modulated ${DESTINATION}
cp -rf surround_control ${DESTINATION}
cp -rf contrast_modulated_no_surround ${DESTINATION}
cp -rf *gilbert* ${DESTINATION}
cp -rf kapadia_experiment ${DESTINATION}

cd ../refactor_gammanet
# conda deactivate
source /media/data/anaconda/etc/profile.d/conda.sh
conda activate py2
bash DEBUG_ALL.sh
cd ../is_exps
conda deactivate
bash DEBUG_ALL.sh
bash RUN_K_EXPS.sh

