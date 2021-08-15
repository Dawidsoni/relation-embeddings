#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/wn18rr_transformer_softmax_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_softmax_reversed_parameter_search_wn18rr'" \
    --gin_bindings "create_knowledge_base_state.model_type = %ModelType.TRANSFORMER_REVERSED_EDGE" \
    --search_config ../search_configs/transformer_softmax_large.json
