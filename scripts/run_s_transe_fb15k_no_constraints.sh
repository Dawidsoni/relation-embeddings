#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_s_transe_training_config.gin \
    --gin_bindings "s_transe/EmbeddingsTransformConfig.constrain_embeddings_norm = False" \
    "s_transe/EmbeddingsTransformConfig.constrain_transformed_embeddings_norm = False" \
    "ExperimentConfig.experiment_name = 's_transe_fb15k_no_constraints'"
