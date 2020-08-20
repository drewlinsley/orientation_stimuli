rm feature_matrix.npz
rm *.joblib

ENCODER=extract_activities_encoder.py
TB_CHANNELS=6
BWC_CHANNELS=12  # 12
TB_EXPDIFF=3
BWC_EXPDIFF=3


# 768, 896; 1280, 1408

# Derive synthetic neurons (train models)
python $ENCODER --responses=orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population
python $ENCODER --responses=orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=bwc_model.joblib --train_moments=bwc_feature_matrix_bwc.npz --channels=$BWC_CHANNELS --exp_diff=$BWC_EXPDIFF --population
# python $ENCODER orientation_probe_outputs 12 feature_select feature_matrix.npz
# python $ENCODER orientation_probe_no_surround_theta_0_outputs 12 feature_select feature_matrix.npz
# python create_both_training_set.py
# python $ENCODER both_outputs 12 feature_select feature_matrix.npz
# python $ENCODER original_gratings_outputs 11 feature_select feature_matrix.npz


# Demonstrate that this approach works with no- and yes-surround images
python $ENCODER --responses=orientation_probe_no_surround_outputs --model_output=orientation_probe_no_surround_outputs_data.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS
python $ENCODER --responses=orientation_probe_outputs --model_output=orientation_probe_outputs_data.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --population
python $ENCODER --responses=surround_control_outputs --model_output=surround_control_outputs.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --population

# Get responses on TB
python $ENCODER --responses=plaid_no_surround_outputs --model_output=plaid_no_surround_outputs_data.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --population
python $ENCODER --responses=plaid_surround_outputs --model_output=plaid_surround_outputs_data.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --population
python $ENCODER --responses=surround_control_outputs --model_output=surround_control_outputs_data.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --population
python $ENCODER --responses=orientation_tilt_outputs --model_output=orientation_tilt_outputs_data.npy --meta_dim=12 --model_file=tb_model.joblib --train_moments=tb_feature_matrix.npz --channels=$TB_CHANNELS --population

# Get responses on BWC
python $ENCODER --responses=contrast_modulated_outputs --model_output=contrast_modulated_outputs_data.npy --meta_dim=14 --model_file=bwc_model.joblib --train_moments=bwc_feature_matrix_bwc.npz --channels=$BWC_CHANNELS --population
python $ENCODER --responses=contrast_modulated_no_surround_outputs --model_output=contrast_modulated_no_surround_outputs_data.npy --meta_dim=14 --model_file=bwc_model.joblib --train_moments=bwc_feature_matrix_bwc.npz --channels=$BWC_CHANNELS --population

# Plot results
python encoding_plot_orientation_responses.py gammanet_full
python encoding_fig4b.py gammanet_full
python encoding_tb_fig2.py gammanet_full
python encoding_bwc_fig4a.py gammanet_full
