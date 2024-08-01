#!/bin/bash
#SBATCH --job-name=temple_rf
#SBATCH --output=std-out/temple_rf_%j.out
#SBATCH --error=std-out/temple_rf_%j.err
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --mem-per-cpu=2G

module --ignore_cache load miniconda
source activate /home/bo.stevens/.conda/envs/rf_env

# Run Python script with input parameters
python3 rf_models.py --field $FIELD --year $YEAR --soil_test $SOIL_TEST --depth $DEPTH --analysis_type $ANALYSIS_TYPE --phosphorus_treatment $PHOSPHORUS_TREATMENT