#!/bin/bash
#SBATCH --job-name=interpolate
#SBATCH --output=std-out/interpolate.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --time=48:00:00
#SBATCH --mail-user=bo.stevens@usda.gov
#SBATCH --mail-type=START,END,FAIL

source /project/akron/Temple/code/spacepy/bin/activate/bin/activate

python interpolate.py