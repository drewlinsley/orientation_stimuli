mkdir responses
rm -rf *outputs
# rsync -r --progress dlinsley@transfer.ccv.brown.edu:/users/dlinsley/scratch/gammanet_neural_data/*outputs responses/  # --exclude=*contrast*
rsync -r --progress dlinsley@ssh.ccv.brown.edu:/users/dlinsley/scratch/gammanet_neural_data/INSILICO* responses/INSILICO/
# rsync -r --progress drew@serrep7.services.brown.edu:/media/data_cifs/cluster_projects/refactor_gammanet/TAE_* responses/TAE/

