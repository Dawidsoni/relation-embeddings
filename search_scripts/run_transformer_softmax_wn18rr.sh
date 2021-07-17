#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/wn18rr_transformer_softmax_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_softmax_parameter_search_wn18rr'" \
    --search_config ../search_configs/transformer_softmax.json
