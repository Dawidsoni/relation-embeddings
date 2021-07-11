#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "NormLossObject.margin = 1.0" \
    "s_transe/EmbeddingsTransformConfig.trainable_transformations_min_iteration = 0" \
    "PiecewiseLinearDecayScheduler.initial_learning_rate = 3e-3" \
    "PiecewiseLinearDecayScheduler.decay_steps = 5800" \
    "PiecewiseLinearDecayScheduler.decay_rate = 0.1" \
    "ExperimentConfig.training_steps = 17_400" \
    "ExperimentConfig.experiment_name = 's_transe_fb15k_no_freeze_margin_1'"
