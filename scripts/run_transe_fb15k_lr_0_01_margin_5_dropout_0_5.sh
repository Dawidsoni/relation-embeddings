#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_transe_training_config.gin \
    --gin_bindings "PiecewiseLinearDecayScheduler.initial_learning_rate = 1e-2" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "SamplingEdgeDataset.batch_size = 10_000" \
    "NormLossObject.margin = 5.0" \
    "s_transe/ConvModelConfig.dropout_rate = 0.5" \
    "ExperimentConfig.experiment_name = 'transe_fb15k_lr_0_01_margin_5_dropout_0_5'"
