#!/bin/bash

# Delete all previous output and error files
rm std-out/*.out
rm std-out/*.err

FIELDS=('16A' '6-12' 'SW16' 'Y8' 'Y10')
YEARS=(2018 2019 2020 2022)
SOIL_TESTS=('DI' 'H3A' 'M3' 'Ols', 'Both')
PHOSPHORUS_TREATMENTS=(1 0)

for FIELD in "${FIELDS[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        for SOIL_TEST in "${SOIL_TESTS[@]}"; do
            for PHOSPHORUS_TREATMENT in "${PHOSPHORUS_TREATMENTS[@]}"; do
                sbatch --export=FIELD="$FIELD",YEAR="$YEAR",SOIL_TEST="$SOIL_TEST",PHOSPHORUS_TREATMENT="$PHOSPHORUS_TREATMENT" run_rf_model.slurm
            done
        done
    done
done
