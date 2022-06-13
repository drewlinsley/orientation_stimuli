MODELS_DIR=linear_models
MOMENTS_DIR=linear_moments
OUTPUTS_DIR=model_outputs
RESPONSE_DIR=../refactor_gammanet/INSILICO_data
mkdir $MODELS_DIR
mkdir $MOMENTS_DIR
mkdir $OUTPUTS_DIR
ENCODER=extract_activities_encoder.py
TB_CHANNELS=6
BWC_CHANNELS=12
PHASE_CHANNELS=6  # 12
TB_EXPDIFF=3
BWC_EXPDIFF=3
PHASE_EXPDIFF=3
MODELS=(
    # INSILICO_BSDS_vgg_gratings_simple_crf
    # INSILICO_BSDS_vgg_gratings_simple_ecrf
    # INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger
    # INSILICO_BSDS_vgg_gratings_simple
    # INSILICO_BSDS_vgg_gratings_horizontal_bigger
    # INSILICO_BSDS_vgg_gratings_horizontal
    # INSILICO_BSDS_vgg_gratings_simple_no_additive
    # INSILICO_BSDS_vgg_gratings_simple_no_additive_no_multiplicative
    # INSILICO_BSDS_vgg_gratings_simple_no_h
    # INSILICO_BSDS_vgg_gratings_simple_no_multiplicative
    # INSILICO_BSDS_vgg_gratings_simple_no_nonnegative
    # INSILICO_BSDS_vgg_gratings_simple_ts_1
    # INSILICO_BSDS_vgg_gratings_simple_untied

    # INSILICO_gammanet_bsds_gratings_3
    INSILICO_gammanet_bsds_gratings_1
    # INSILICO_gammanet_bsds_gratings_2
)

len=${#MODELS[@]}
echo $len
for (( i=0; i<len; i++ ))
do
    MODEL_NAME=${MODELS[$i]}
    echo $MODEL_NAME
    TB_MODEL=linear_models/${MODEL_NAME}_tb_model.joblib
    TB_CONV_MODEL=linear_models/${MODEL_NAME}_conv2_2_tb_model.joblib
    GILBERT_MODEL=linear_models/${MODEL_NAME}_gilbert_model.joblib
    GILBERT_CONV_MODEL=linear_models/${MODEL_NAME}_conv2_2_gilbert_model.joblib
    FULLFIELD_FF_CONV_MODEL=linear_models/${MODEL_NAME}_fb_tb_model.joblib
    FULLFIELD_FB_CONV_MODEL=linear_models/${MODEL_NAME}_ff_tb_model.joblib
    BWC_MODEL=linear_models/${MODEL_NAME}_bwc_model.joblib
    PHASE_MODEL=linear_models/${MODEL_NAME}_phase_model.joblib
    TB_MOMENTS=linear_moments/${MODEL_NAME}_tb_feature_matrix.npz
    TB_CONV_MOMENTS=linear_moments/${MODEL_NAME}_conv2_2_tb_feature_matrix.npz
    GILBERT_MOMENTS=linear_moments/${MODEL_NAME}_gilbert_feature_matrix.npz
    GILBERT_CONV_MOMENTS=linear_moments/${MODEL_NAME}_conv2_2_gilbert_feature_matrix.npz
    FULLFIELD_FF_CONV_MOMENTS=linear_moments/${MODEL_NAME}_ff_tb_feature_matrix.npz
    FULLFIELD_FB_CONV_MOMENTS=linear_moments/${MODEL_NAME}_fb_tb_feature_matrix.npz
    BWC_MOMENTS=linear_moments/${MODEL_NAME}_bwc_feature_matrix.npz
    PHASE_MOMENTS=linear_moments/${MODEL_NAME}_phase_feature_matrix.npz

    # Optimize synthetic neurons
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_plaid_no_surround --meta_dim=12 --train_model --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --exp_diff=$BWC_EXPDIFF --population

    # Get responses on BWC
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_contrast_test_no_surround --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy --meta_dim=14 --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --population

    # Plot results
    python encoding_bwc_fig4a.py $MODEL_NAME ${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy
    echo "END"
done

