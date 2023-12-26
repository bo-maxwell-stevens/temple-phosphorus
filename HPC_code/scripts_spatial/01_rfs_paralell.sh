#!/bin/bash

# Delete all previous output and error files
rm std-out/*.out
rm std-out/*.err

FIELDS=('16A' '6-12' 'SW16' 'Y8' 'Y10')
YEARS=(2018 2019 2020 2022)
ZONES=('Zone_1' 'Zone_2' 'Zone_3' 'Zone_4')
PHOSPHORUS_TREATMENTS=(1 0)

for FIELD in "${FIELDS[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        for ZONE in "${ZONES[@]}"; do
            for PHOSPHORUS_TREATMENT in "${PHOSPHORUS_TREATMENTS[@]}"; do
                sbatch --export=FIELD="$FIELD",YEAR="$YEAR",ZONE="$ZONE",PHOSPHORUS_TREATMENT="$PHOSPHORUS_TREATMENT" run_rf_model.slurm
            done
        done
    done
done
