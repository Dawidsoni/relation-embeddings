#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_convkb_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'convkb_wn18rr_baseline'"
