#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_transe_training_config.gin \
    --gin_bindings "create_learning_rate_schedule.initial_learning_rate = 3e-3" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "Dataset.batch_size = 10_000" \
    "ExperimentConfig.experiment_name = 'transe_fb15k_baseline'"
