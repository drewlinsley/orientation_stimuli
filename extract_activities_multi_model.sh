rm feature_matrix.npz
rm *.joblib

MODELS_DIR=linear_models_multi
MOMENTS_DIR=linear_moments_multi
OUTPUTS_DIR=model_outputs
RESPONSE_DIR=responses
mkdir $MODELS_DIR
mkdir $MOMENTS_DIR
mkdir $OUTPUTS_DIR
MODEL_NAME=gammanet_full
ENCODER=extract_activities_encoder_multi_model.py
TB_CHANNELS=6
TB_EXPDIFF=3
TB_MODEL=linear_models/${gammanet_full}tb_model.joblib
TB_MOMENTS=linear_moments/${gammanet_full}tb_feature_matrix.npz

# Derive synthetic neurons (train models)
echo ${RESPONSE_DIR}/orientation_probe_no_surround_outputs
python $ENCODER --responses=${RESPONSE_DIR}/orientation_probe_no_surround_outputs --meta_dim=12 --train_model --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population
