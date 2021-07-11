#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "PiecewiseLinearDecayScheduler.initial_learning_rate = 1e-2" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "SamplingEdgeDataset.batch_size = 3000" \
    "NormLossObject.margin = 7.0" \
    "transe/ConvModelConfig.dropout_rate = 0.5" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_margin_7_dropout_0_5'"
