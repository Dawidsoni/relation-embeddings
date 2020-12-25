#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "create_learning_rate_schedule.initial_learning_rate = 1e-2" \
    "create_learning_rate_schedule.decay_steps = 3500" \
    "create_learning_rate_schedule.decay_rate = 0.1" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "Dataset.batch_size = 3000" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_exp_decay_0_01'"
