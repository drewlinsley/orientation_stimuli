#####
# 0. Generate stimuli
bash generate_neural_stim.sh

# 1. Run simulations
- bash refactor_gammanet/in_silico_sims.sh 
# 2. In this directory run linear models for in silico recordings
- bash extract_insilico.sh
# 3. Plot fits/perf over time
- python summarize_results.py
# 4. Plot a comparison of multiple models
- python plot_all_model_perfs.py
# 5. Run functional connectivity study
- Navigate to refactor_gammanet_connectivity for the readme there

