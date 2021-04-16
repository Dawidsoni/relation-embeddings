#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_s_transe_training_config.gin \
    --gin_bindings "_create_learning_rate_schedule.initial_learning_rate = 1e-4" \
    "ExperimentConfig.training_steps = 29000" \
    "ExperimentConfig.experiment_name = 's_transe_wn18rr_lr_0_0001'"
