#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "NormLossObject.margin = 5.0" \
    "s_transe/EmbeddingsTransformConfig.trainable_transformations_min_iteration = 5800" \
    "PiecewiseLinearDecayScheduler.initial_learning_rate = 3e-3" \
    "PiecewiseLinearDecayScheduler.decay_steps = 5800" \
    "PiecewiseLinearDecayScheduler.decay_rate = 0.03" \
    "ExperimentConfig.experiment_name = 's_transe_fb15k_freeze_5800_margin_5_lr_decay_0_03'"
