#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/fb15k_pretrained_convkb_training_config.gin \
    --gin_bindings "convkb/ConvModelConfig.normalize_embeddings = False" \
    "ExperimentConfig.experiment_name = 'convkb_fb15k_pretrained'"
