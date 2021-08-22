#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/fb15k_transformer_softmax_all_neighbours_training_config.gin \
    --search_config ../search_configs/transformer_softmax_finetune.json