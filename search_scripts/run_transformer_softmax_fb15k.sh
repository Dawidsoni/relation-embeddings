#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/fb15k_transformer_softmax_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_softmax_parameter_search_fb15k'" \
    --search_config ../search_configs/transformer_softmax.json
