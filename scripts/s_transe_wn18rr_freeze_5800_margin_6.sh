#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_s_transe_training_config.gin \
    --gin_bindings "NormLossObject.margin = 6.0" \
    "s_transe/EmbeddingsTransformConfig.trainable_transformations_min_iteration = 5800" \
    "_create_learning_rate_schedule.initial_learning_rate = 1e-2" \
    "_create_learning_rate_schedule.decay_steps = 5800" \
    "ExperimentConfig.experiment_name = 's_transe_wn18rr_freeze_5800_margin_6'"
