#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_transe_training_config.gin \
    --gin_bindings "_create_learning_rate_schedule.initial_learning_rate = 1e-2" \
    "ExperimentConfig.steps_per_evaluation = 50" \
    "SamplingEdgeDataset.batch_size = 3000" \
    "NormLossObject.margin = 6.0" \
    "transe/ConvModelConfig.dropout_rate = 0.0" \
    "_create_embeddings_config.entity_embeddings_path = '/pio/scratch/1/i279743/models/transe_wn18rr_margin_6_1608882960/entity_embeddings.npy'" \
    "_create_embeddings_config.relation_embeddings_path = '/pio/scratch/1/i279743/models/transe_wn18rr_margin_6_1608882960/relation_embeddings.npy'" \
    "ExperimentConfig.training_steps = 2" \
    "ExperimentConfig.experiment_name = 'transe_wn18rr_pretrained'"
