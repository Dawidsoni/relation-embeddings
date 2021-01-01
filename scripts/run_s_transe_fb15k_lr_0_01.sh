#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "create_learning_rate_schedule.initial_learning_rate = 1e-2" \
    "ExperimentConfig.training_steps = 14500" \
    "ExperimentConfig.experiment_name = 's_transe_fb15k_lr_0_01'"
