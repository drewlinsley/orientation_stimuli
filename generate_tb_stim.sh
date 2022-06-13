DESTINATION=../refactor_gammanet
rm -rf ../refactor_gammanet/INSILICO_data_all_models/INSILICO_data_all_models_gammanet_bsds_model_1376000
rm -rf orientation_probe_no_surround
rm -rf orientation_tilt
# rm -rf orientation_probe
rm -rf plaid_no_surround
rm -rf surround_search
rm -rf plaid_surround

python tb_stim_wrapper.py 1 1 1 orientation_probe_no_surround
python tb_stim_wrapper.py 1 1 1 orientation_tilt
# python tb_stim_wrapper.py 1 1 1 orientation_probe
python tb_stim_wrapper.py 1 1 1 plaid_surround
python tb_stim_wrapper.py 1 1 1 plaid_no_surround
python tb_stim_wrapper.py 1 1 1 surround_search

rm -rf ${DESTINATION}/orientation_probe*
rm -rf ${DESTINATION}/orientation_tilt
rm -rf ${DESTINATION}/plaid_*surround
rm -rf ${DESTINATION}/surround_search

cp -rf orientation_probe* ${DESTINATION}
cp -rf orientation_tilt ${DESTINATION}
cp -rf plaid_*surround ${DESTINATION}
cp -rf surround_search ${DESTINATION}

cd ../refactor_gammanet
# conda deactivate
source /media/data/anaconda/etc/profile.d/conda.sh
conda activate py2
bash DEBUG_TB_ONLY.sh
cd ../is_exps
conda deactivate
bash DEBUG_TB.sh

