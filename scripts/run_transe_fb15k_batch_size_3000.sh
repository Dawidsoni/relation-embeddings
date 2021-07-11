#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_transe_training_config.gin \
    --gin_bindings "PiecewiseLinearDecayScheduler.initial_learning_rate = 3e-3" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "SamplingEdgeDataset.batch_size = 3000" \
    "ExperimentConfig.experiment_name = 'transe_fb15k_batch_size_3000'"
