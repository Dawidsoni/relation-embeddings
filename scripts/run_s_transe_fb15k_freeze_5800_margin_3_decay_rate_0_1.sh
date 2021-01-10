#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "LossObject.norm_metric_margin = 3.0" \
    "s_transe/EmbeddingsTransformConfig.trainable_transformations_min_iteration = 5800" \
    "create_learning_rate_schedule.initial_learning_rate = 3e-3" \
    "create_learning_rate_schedule.decay_steps = 5800" \
    "create_learning_rate_schedule.decay_rate = 0.1" \
    "ExperimentConfig.training_steps = 17_400" \
    "ExperimentConfig.experiment_name = 's_transe_fb15k_freeze_5800_margin_3_decay_rate_0_1'"
