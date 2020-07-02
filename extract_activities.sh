rm feature_matrix.npz
rm *.joblib

# Derive synthetic neurons (train models)
python extract_activities.py orientation_probe_no_surround_outputs 12 feature_select feature_matrix.npz
# python extract_activities.py orientation_probe_outputs 12 feature_select feature_matrix.npz
# python extract_activities.py orientation_probe_no_surround_theta_0_outputs 12 feature_select feature_matrix.npz
# python extract_activities.py both 12 feature_select feature_matrix.npz
# python extract_activities.py original_gratings_outputs 11 feature_select feature_matrix.npz



# Demonstrate that this approach works with no- and yes-surround images
python extract_activities.py orientation_probe_no_surround_outputs 12 activity orientation_probe_no_surround_outputs_data.npy feature_matrix.npz
python extract_activities.py orientation_probe_outputs 12 activity orientation_probe_outputs_data.npy feature_matrix.npz

# Get Test data
python extract_activities.py plaid_no_surround_outputs 12 activity plaid_no_surround_outputs_data.npy feature_matrix.npz
python extract_activities.py orientation_tilt_outputs 12 activity orientation_tilt_outputs_data.npy feature_matrix.npz
python extract_activities.py plaid_surround_outputs 12 activity plaid_surround_outputs_data.npy feature_matrix.npz
python extract_activities.py contrast_modulated_outputs 14 activity contrast_modulated_outputs_data.npy feature_matrix.npz
python extract_activities.py contrast_modulated_no_surround_outputs 14 activity contrast_modulated_no_surround_outputs_data.npy feature_matrix.npz save_images
 
python tb_fig4b.py
python tb_fig2.py
python bwc_fig4a.py