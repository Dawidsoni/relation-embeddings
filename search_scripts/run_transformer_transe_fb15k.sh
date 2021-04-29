#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/wn18rr_transformer_transe_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_transe_parameter_search'" \
    --search_config ../search_configs/transformer_transe_wn18rr.json

