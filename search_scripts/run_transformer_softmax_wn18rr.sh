#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/wn18rr_transformer_softmax_training_config.gin \
    --search_config ../search_configs/transformer_softmax_wn18rr.json
