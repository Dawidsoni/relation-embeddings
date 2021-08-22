#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/fb15k_transformer_softmax_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_reversed_edge_parameter_search_fb15k'" \
    --gin_bindings "create_knowledge_base_state.model_type = %ModelType.TRANSFORMER_REVERSED_EDGE" \
    --search_config ../search_configs/transformer_softmax_finetune.json
