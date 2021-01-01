#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_convkb_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'convkb_fb15k_baseline'"
