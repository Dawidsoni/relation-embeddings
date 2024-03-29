#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_pretrained_convkb_training_config.gin \
    --gin_bindings "convkb/ConvModelConfig.normalize_embeddings = False" \
    "ExperimentConfig.experiment_name = 'convkb_wn18rr_pretrained'"
