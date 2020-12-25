#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "create_learning_rate_schedule.initial_learning_rate = 1e-2" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "Dataset.batch_size = 3000" \
    "LossObject.norm_metric_margin = 6.0" \
    "transe/ModelConfig.dropout_rate = 0.5" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_margin_6_dropout_0_5'"
