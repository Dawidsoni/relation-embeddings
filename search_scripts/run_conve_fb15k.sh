#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/fb15k_conve_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'conve_parameter_search_fb15k'" \
    --search_config ../search_configs/conve.json
