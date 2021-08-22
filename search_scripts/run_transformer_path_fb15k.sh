#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/fb15k_transformer_path_training_config.gin \
    --search_config ../search_configs/transformer_path_finetune.json
