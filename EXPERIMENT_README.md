- IN SILICO RECORDINGS
0. (P7):
`cd /media/data_cifs/projects/prj_neural_circuits/refactor_gammanet`
`bash new_bsds_training.sh`
`bash new_bsds_eval.sh`
1. (HERE) Create datasets here: `bash generate_neural_stim.sh`
2. (HERE) Transfer data to CCV: `bash transfer_responses_to_ccv.sh`
3. (CCV neural_stims/) Transfer data to pnodes: `transfer_images_to_isilon.sh`
4. (p7):
`cd /media/data_cifs/projects/prj_neural_circuits/refactor_gammanet`
`bash neural_stim_data_all_models.sh`
5. (CCV neural_stims/) Transfer data to CCV: `bash get_results_from_isilon.sh
6. (HERE) DL data: `bash dl_from_ccv.sh`
7. (HERE) Organize data manually in responses/ :-(
8. (HERE) `bash INSILICO_extract_activities.sh`
9. (HERE) `python plot_insilico.py`

- eCRF INFERENCE
0. (HERE) `bash today_transfer.sh`
1. (CCV new_circuit_responses)
2. (P7):
`cd /media/data_cifs/projects/prj_neural_circuits/refactor_gammanet_p7`
Create circuit files
Run the circuit files
3. Plot the data there
4. DL the data
5. (HERE) Plot the tuning here

- Dreaming

