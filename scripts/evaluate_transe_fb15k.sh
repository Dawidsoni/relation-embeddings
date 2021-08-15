#!/usr/bin/env bash

python3 ../src/evaluate_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 's_transe_fb15k_baseline'" \
    "load_and_set_model_weights.model_weights_path = '/Users/dawidwegner/Work/ms-thesis/data/models/s_transe_fb15k_baseline_1609107746/saved_weights.tf'"
