# conda deactivate

MODELS_DIR=linear_models
MOMENTS_DIR=linear_moments
OUTPUTS_DIR=model_outputs
RESPONSE_DIR=../refactor_gammanet/INSILICO_data_all_models
mkdir $MODELS_DIR
mkdir $MOMENTS_DIR
mkdir $OUTPUTS_DIR
ENCODER=optim_encoder.py
TB_CHANNELS=6
KAPADIA_CHANNELS=6
BWC_CHANNELS=12  # 12
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
    python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_orientation_search --meta_dim=12 --train_model --model_file=$TB_MODEL --train_moments=$TB_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population  # --meta_col=10  #  --debug

    # Demonstrate that this approach works with on- and off-surround images
done

