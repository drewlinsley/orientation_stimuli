

MODELS_DIR=linear_models
MOMENTS_DIR=linear_moments
OUTPUTS_DIR=model_outputs
RESPONSE_DIR=../refactor_gammanet/INSILICO_data
mkdir $MODELS_DIR
mkdir $MOMENTS_DIR
mkdir $OUTPUTS_DIR
ENCODER=extract_activities_encoder.py
TB_CHANNELS=180
BWC_CHANNELS=6  # 12
PHASE_CHANNELS=6  # 12
TB_EXPDIFF=179  # 5
BWC_EXPDIFF=5
PHASE_EXPDIFF=5  # 5
MODEL_NAME=INSILICO_data_gammanet_bsds_gratings_model_268000
GILBERT_MODEL=linear_models/${MODEL_NAME}_gilbert_model.joblib
GILBERT_CONV_MODEL=linear_models/${MODEL_NAME}_conv2_2_gilbert_model.joblib
GILBERT_MOMENTS=linear_moments/${MODEL_NAME}_gilbert_feature_matrix.npz
GILBERT_CONV_MOMENTS=linear_moments/${MODEL_NAME}_conv2_2_gilbert_feature_matrix.npz


cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_train_flip
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_train_flip --meta_dim=12 --train_model --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population --meta_col=2
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_train_flip --meta_dim=12 --train_model --model_file=$GILBERT_CONV_MODEL --train_moments=$GILBERT_CONV_MOMENTS --channels=$TB_CHANNELS --exp_diff=$TB_EXPDIFF --population --meta_col=2 --extract_key=conv2_2

cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_train_flip
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_train_flip --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_train_flip_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2
# python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_repulse_flip --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_repulse_flip_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2

cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_offset_flip
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_offset_flip --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_offset_flip_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2

cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_offset
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_offset --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_offset_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2

cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_flip
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_flip --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_flip_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2
cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_right_flip
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_right_flip --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_right_flip_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2
cp gilbert_angelluci_right/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_left_flip
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci_left_flip --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_left_flip_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2

cp gilbert_angelluci/test/metadata/1.npy ${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci
python $ENCODER --responses=${RESPONSE_DIR}/${MODEL_NAME}/${MODEL_NAME}_gilbert_angelluci --model_output=${OUTPUTS_DIR}/${MODEL_NAME}_gilbert_angelluci_outputs_data.npy --meta_dim=12 --model_file=$GILBERT_MODEL --train_moments=$GILBERT_MOMENTS --channels=$TB_CHANNELS --population --meta_col=2


