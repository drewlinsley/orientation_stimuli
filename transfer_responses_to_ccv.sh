
# scp model_outputs/gammanet_full_orientation_probe_contrast_0.* dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_contrast_modulated_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_plaid_surround_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_plaid_no_surround_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_tilt_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_no_surround_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_contrast_modulated_no_surround_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_phase_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_full_field_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

###
# scp model_outputs/gammanet_full_orientation_probe_contrast_06_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_contrast_12_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_contrast_25_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_contrast_50_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# scp model_outputs/gammanet_full_orientation_probe_contrast_75_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# # Create flipped contrast responses
# python flip_contrast_response_order.py
# scp model_outputs/gammanet_full_contrast_modulated_no_surround_outputs_data_flipped.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# # Some extra models
# scp linear_models/tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
# scp linear_moments/tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# # Full field models and data
# scp model_outputs/gammanet_full_conv2_2_orientation_probe_full_field_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
# scp linear_models/conv2_2* dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# Full field models and data for full
scp model_outputs/INSILICO_BSDS_vgg_gratings_simpleconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simplefull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simpletb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_gilbert_angelluci_outputs_train_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_gilbert_angelluci_outputs_data.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_gilbert_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_conv2_2_gilbert_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_gilbert_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# Full field models and data for td- and h-only
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_hconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_hfull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_htb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_h_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_h_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontal_biggerconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontal_biggerfull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontal_biggertb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_bigger_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_bigger_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_horizontal_bigger_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_horizontal_bigger_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontalconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontalfull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontaltb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_horizontal_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_horizontal_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

#eCRF
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_ecrfconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_ecrffull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_ecrftb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_ecrf_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_ecrf_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

#eCRF_bigger
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_ecrf_biggerconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_ecrf_biggerfull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_ecrf_biggertb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_ecrf_bigger_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# TD
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_hconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_hfull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_htb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_h_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_h_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_h_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/


# H
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontalconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontalfull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_horizontaltb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_horizontal_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_horizontal_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_horizontal_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_horizontal_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_horizontal_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/


#CRF
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_crfconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_crffull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_crftb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_crf_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_crf_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_crf_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_crf_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_crf_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_crf_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/


# nomultiplicative
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_multiplicativeconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_multiplicativefull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_multiplicativetb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_multiplicative_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_multiplicative_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_multiplicative_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_multiplicative_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_multiplicative_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_multiplicative_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/

# nosub
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_additiveconv2_2_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_additivefull_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp model_outputs/INSILICO_BSDS_vgg_gratings_simple_no_additivetb_full_field_outputs.npy dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_additive_ff_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_additive_fb_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_additive_ff_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_moments/INSILICO_BSDS_vgg_gratings_simple_no_additive_fb_tb_feature_matrix.npz dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_additive_conv2_2_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/
scp linear_models/INSILICO_BSDS_vgg_gratings_simple_no_additive_tb_model.joblib dlinsley@transfer.ccv.brown.edu:/users/dlinsley/new_circuit_responses/


# 15885239




# h_connectome
# bash run_circuits.sh

# crf_connectome
# bash run_circuits_crf.sh

# # crf_train
# # CUDA_VISIBLE_DEVICES=6 python run_job.py --experiment=BSDS500_combos_100_no_aux_jigsaw_crf_rand_init --model=BSDS_vgg_cheap_deepest_final_simple_no_imagenet --no_db --train=BSDS500_100_jk --val=BSDS500_100_jk --placeholder_test=BSDS500_test_padded

# lightness
# bash exp_8_multi_lightness.sh

# tilts
# bash exp_7_multi_tilts.sh

###
# bash neural_stim_INFLUENCE_MAPPING.sh
# bash neural_stim_data_all_models_ECRF.sh

# Download all the illusions
# Plot and download all the connectomes
# Add a new analysis to the connnectome where you take a unit from near surround and plot how it changes w/ perturb strength
# Download the influence mapping files and compute the influence mapping score


