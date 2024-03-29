#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/wn18rr_conve_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'conve_parameter_search_wn18rr'" \
    --search_config ../search_configs/conve.json
