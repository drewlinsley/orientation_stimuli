rm -rf *outputs
rsync -r --progress dlinsley@transfer.ccv.brown.edu:/users/dlinsley/scratch/*outputs .  # --exclude=*contrast*
rsync -r --progress dlinsley@transfer.ccv.brown.edu:/users/dlinsley/scratch/INSILICO* INSILICO/
rsync -r --progress drew@serrep7.services.brown.edu:/media/data_cifs/cluster_projects/refactor_gammanet/TAE_* TAE/
