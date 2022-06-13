# conda deactivate

MODELS_DIR=linear_models
MOMENTS_DIR=linear_moments
OUTPUTS_DIR=model_outputs
RESPONSE_DIR=../refactor_gammanet/INSILICO_data_all_models
mkdir $MODELS_DIR
mkdir $MOMENTS_DIR
mkdir $OUTPUTS_DIR
ENCODER=extract_activities_encoder.py
TB_CHANNELS=6
KAPADIA_CHANNELS=6
BWC_CHANNELS=12
TB_EXPDIFF=3
KAPADIA_EXPDIFF=3
BWC_EXPDIFF=1

for MODEL_NAME in `ls -t1 $RESPONSE_DIR`
do
    MODEL_NAME=`echo $MODEL_NAME | sed 's/..\/refactor_gammanet\/models\///g'`
    MODEL_NAME=`echo $MODEL_NAME | sed 's/.py//g'`
    echo $MODEL_NAME
    TB_MODEL=${MODELS_DIR}/${MODEL_NAME}_tb_model.joblib
    TB_CONV_MODEL=${MODELS_DIR}/${MODEL_NAME}_conv2_2_tb_model.joblib
    GILBERT_MODEL=${MODELS_DIR}/${MODEL_NAME}_gilbert_model.joblib
    GILBERT_CONV_MODEL=${MODELS_DIR}/${MODEL_NAME}_conv2_2_gilbert_model.joblib
    BWC_MODEL=${MODELS_DIR}/${MODEL_NAME}_bwc_model.joblib
    KAPADIA_MODEL=${MODELS_DIR}/${MODEL_NAME}_kapadia_model.joblib

    TB_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_tb_feature_matrix.npz
    TB_CONV_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_conv2_2_tb_feature_matrix.npz
    GILBERT_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_gilbert_feature_matrix.npz
    GILBERT_CONV_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_conv2_2_gilbert_feature_matrix.npz
    FULLFIELD_FF_CONV_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_ff_tb_feature_matrix.npz
    FULLFIELD_FB_CONV_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_fb_tb_feature_matrix.npz
    BWC_MOMENTS=${MOMENTS_DIR}/${MODEL_NAME}_bwc_feature_matrix.npz
    KAPADIA_MOMENTS=${MODELS_DIR}/${MODEL_NAME}_kapadia_feature_matrix.npz

    # Optimize synthetic neurons
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_orientation_test_no_surround --meta_dim=12 --train_model --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population  #  --debug
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_orientation_test_no_surround --meta_dim=12 --train_model --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --exp_diff=$BWC_EXPDIFF --population # --extract_key=conv2_2  #  --debug

    # Get responses on TB
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_plaid_surround --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_plaid_surround_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population  #  --debug
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_plaid_no_surround --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_plaid_no_surround_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population  #  --debug
    # python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_surround_search --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_surround_search_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population  #  --debug
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_orientation_test --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_orientation_tilt --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_orientation_tilt_outputs_data.npy --meta_dim=12 --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --population

    # Get responses on BWC
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_contrast_test_no_surround --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy --meta_dim=14 --model_file=$BWC_MODEL --train_moments=$BWC_MOMENTS --channels=$BWC_CHANNELS --population  # --debug

    # Plot results
    echo $MODEL_NAME ${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy
    # python encoding_plot_orientation_responses.py $MODEL_NAME ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_no_surround_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_probe_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_surround_control_outputs.npy
    # python encoding_fig4b.py $MODEL_NAME ${OUTPUTS_DIR}/${MODEL_NAME}_plaid_no_surround_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_plaid_surround_outputs_data.npy
    # python encoding_tb_fig2.py $MODEL_NAME ${OUTPUTS_DIR}/${MODEL_NAME}_plaid_no_surround_outputs_data.npy ${OUTPUTS_DIR}/${MODEL_NAME}_orientation_tilt_outputs_data.npy
    python encoding_bwc_fig4a.py $MODEL_NAME ${OUTPUTS_DIR}/${MODEL_NAME}_contrast_modulated_no_surround_outputs_data.npy
    echo "END"
done

