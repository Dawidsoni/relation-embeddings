#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "NormLossObject.margin = 3.0" \
    "s_transe/EmbeddingsTransformConfig.trainable_transformations_min_iteration = 5800" \
    "_create_learning_rate_schedule.initial_learning_rate = 3e-3" \
    "_create_learning_rate_schedule.decay_steps = 5800" \
    "_create_learning_rate_schedule.decay_rate = 0.3" \
    "ExperimentConfig.training_steps = 29_000" \
    "ExperimentConfig.experiment_name = 's_transe_fb15k_freeze_5800_margin_3'"