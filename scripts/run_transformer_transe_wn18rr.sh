#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transformer_transe_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_transe_wn18rr_baseline'"
