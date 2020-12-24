#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "create_learning_rate_schedule.initial_learning_rate = 1e-3" \
    "create_learning_rate_schedule.decay_steps = 10_000" \
    "create_learning_rate_schedule.decay_rate = 0.5" \
    "ExperimentConfig.steps_per_evaluation = 500" \
    "Dataset.batch_size = 256" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_batch256'"
