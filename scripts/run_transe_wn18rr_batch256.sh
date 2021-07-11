#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "PiecewiseLinearDecayScheduler.initial_learning_rate = 1e-3" \
    "PiecewiseLinearDecayScheduler.decay_steps = 10_000" \
    "PiecewiseLinearDecayScheduler.decay_rate = 0.5" \
    "ExperimentConfig.steps_per_evaluation = 500" \
    "SamplingEdgeDataset.batch_size = 256" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_batch256'"
