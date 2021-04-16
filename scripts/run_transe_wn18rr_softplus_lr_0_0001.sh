#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "_create_learning_rate_schedule.initial_learning_rate = 1e-4" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "Dataset.batch_size = 3000" \
    "create_knowledge_base_state.loss_type = %LossType.SOFTPLUS" \
    "transe/ConvModelConfig.include_reduce_dim_layer = True" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_softplus_lr_0_0001'"
