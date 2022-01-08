rm feature_matrix.npz
rm *.joblib

MODELS_DIR=linear_models
MOMENTS_DIR=linear_moments
OUTPUTS_DIR=model_outputs
RESPONSE_DIR=../refactor_gammanet/INSILICO_data
mkdir $MODELS_DIR
mkdir $MOMENTS_DIR
mkdir $OUTPUTS_DIR
MODEL_NAME=gammanet_full
ENCODER=extract_activities_encoder.py
TB_CHANNELS=6
BWC_CHANNELS=12  # 12
TILT_CHANNELS=10  # 12
TB_EXPDIFF=3
BWC_EXPDIFF=3
TB_MODEL=linear_models/${gammanet_full}tb_model.joblib
TB_CONV_MODEL=linear_models/${gammanet_full}conv2_2_tb_model.joblib
# TB_CONTRAST_MODEL=linear_models/${gammanet_full}tb_contrast_model.joblib
TB_CONTRAST_MODEL=linear_models/${gammanet_full}tb_model.joblib
BWC_MODEL=linear_models/${gammanet_full}bwc_model.joblib
PHASE_MODEL=linear_models/${gammanet_full}bwc_model.joblib
TILT_MODEL=linear_models/${gammanet_full}tilt_model.joblib
TB_MOMENTS=linear_moments/${gammanet_full}tb_feature_matrix.npz
TB_CONV_MOMENTS=linear_models/${gammanet_full}conv2_2_tb_feature_matrix.npz

# TB_CONTRAST_MOMENTS=linear_moments/${gammanet_full}tb_contrast_feature_matrix.npz
TB_CONTRAST_MOMENTS=linear_moments/${gammanet_full}tb_feature_matrix.npz
BWC_MOMENTS=linear_moments/${gammanet_full}bwc_feature_matrix.npz
PHASE_MOMENTS=linear_moments/${gammanet_full}bwc_feature_matrix.npz
TILT_MOMENTS=linear_moments/${gammanet_full}tilt_feature_matrix.npz

# 768, 896; 1280, 1408

# Derive synthetic neurons (train models)
echo ${RESPONSE_DIR}/orientation_probe_no_surround_outputs
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_full_field_outputs --extract_key=conv2_2 --meta_dim=12 --train_model --model_file=$TB_CONV_MODEL --train_moments=$TB_CONV_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population


# python $ENCODER --responses=${RESPONSE_DIR}/phase_modulated_outputs --meta_dim=12 --train_model --model_file=$PHASE_MODEL --train_moments=$PHASE_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population


# Demonstrate that this approach works with no- and yes-surround images
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_no_surround_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_no_surround_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/surround_control_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_surround_control_outputs.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_full_field_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_full_field_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population

# Show that it works on full field stimuli
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_full_field_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_conv2_2_orientation_probe_full_field_outputs_data.npy --meta_dim=12 --model_file=$TB_CONV_MODEL --train_moments=$TB_CONV_MOMENTS --channels=$TB_CHANNELS --population --extract_key=conv2_2

# # Also that surround suppression changes with different contrasts
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_06_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_0.06_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_12_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_0.12_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_25_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_0.25_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_50_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_0.50_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_75_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_0.75_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population


# # Get a contrast model
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_0.75_outputs --meta_dim=12 --train_model --model_file=$TB_CONTRAST_MODEL --train_moments=$TB_CONTRAST_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population

# Now check responses from the different contrast bins
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_06_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_06_outputs_data.npy --meta_dim=12 --model_file=$TB_CONTRAST_MODEL --train_moments=$TB_CONTRAST_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_12_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_12_outputs_data.npy --meta_dim=12 --model_file=$TB_CONTRAST_MODEL --train_moments=$TB_CONTRAST_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_25_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_25_outputs_data.npy --meta_dim=12 --model_file=$TB_CONTRAST_MODEL --train_moments=$TB_CONTRAST_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_50_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_50_outputs_data.npy --meta_dim=12 --model_file=$TB_CONTRAST_MODEL --train_moments=$TB_CONTRAST_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_contrast_75_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_75_outputs_data.npy --meta_dim=12 --model_file=$TB_CONTRAST_MODEL --train_moments=$TB_CONTRAST_MOMENTS --channels=$TB_CHANNELS --population

# Get responses on TB
python $ENCODER --responses=${RESPONSE_DIR}/plaid_no_surround_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_plaid_no_surround_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/plaid_surround_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_plaid_surround_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/surround_control_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_surround_control_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_tilt_retrained_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_tilt_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population

# Train model and get responses on BWC
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --exp_diff=$BWC_EXPDIFF --population --save_images
# python $ENCODER --responses=${RESPONSE_DIR}/contrast_modulated_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_outputs_data.npy --meta_dim=14 --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --population
python $ENCODER --responses=${RESPONSE_DIR}/contrast_modulated_no_surround_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy --meta_dim=14 --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --population --save_images

# Train model and get tilt responses
# python orientation_tilt_illusion.py --responses=${RESPONSE_DIR}/orientation_tilt_retrained_outputs --meta_dim=12 --train_model --model_file=$TILT_MODEL --train_moments=$TILT_MOMENTS --channels=$TILT_CHANNELS --exp_diff=$BWC_EXPDIFF
# python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=$TILT_MODEL --train_moments=$TILT_MOMENTS --channels=$TILT_CHANNELS --exp_diff=$TB_EXPDIFF --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=$TILT_MODEL --train_moments=$TILT_MOMENTS --channels=$BWC_CHANNELS --exp_diff=$TB_EXPDIFF --population
python $ENCODER --responses=${RESPONSE_DIR}/orientation_tilt_retrained_outputs --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_tilt_illusion_outputs_data.npy --meta_dim=12 --model_file=$TILT_MODEL --train_moments=$TILT_MOMENTS --channels=$BWC_CHANNELS --population

# Plot results
python encoding_plot_orientation_responses.py gammanet_full ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_no_surround_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_surround_control_outputs.npy
python encoding_plot_orientation_contrast_responses.py ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_contrast_0.06_outputs_data.npy
python encoding_fig4b.py gammanet_full ${OUTPUTS_DIR}/${MODEL_NAME}_plaid_no_surround_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_plaid_surround_outputs_data.npy
python encoding_tb_fig2.py gammanet_full ${OUTPUTS_DIR}/${MODEL_NAME}_plaid_no_surround_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_tilt_outputs_data.npy
python encoding_bwc_fig4a.py gammanet_full ${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy
# python encoding_plot_tilt_illusion.py gammanet_full ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_tilt_illusion_outputs_data.npy
