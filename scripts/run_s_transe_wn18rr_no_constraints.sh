#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_s_transe_training_config.gin \
    --gin_bindings "s_transe/ModelConfig.normalize_embeddings = False" \
    "s_transe/EmbeddingsTransformConfig.constrain_transformed_embeddings_norm = False" \
    "ExperimentConfig.experiment_name = 's_transe_wn18rr_no_constraints'"

