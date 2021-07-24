#!/usr/bin/env bash

python3 ../src/run_parameter_search.py \
    --gin_configs ../configs/fb15k_transformer_softmax_training_config.gin \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_softmax_all_neighbours_large_parameter_search_fb15k'" \
    --gin_bindings "create_knowledge_base_state.model_type = %ModelType.TRANSFORMER_SOFTMAX_ALL_NEIGHBOURS" \
    --gin_bindings "MaskedEntityAllNeighbours.data_directory = '../data/FB15k-237'" \
    --gin_bindings "MaskedEntityAllNeighbours.batch_size = 512" \
    --gin_bindings "MaskedEntityAllNeighbours.filter_repeated_samples = True" \
    --gin_bindings "ExperimentConfig.experiment_name = 'transformer_softmax_all_neighbours_large_filtered_parameter_search_fb15k'" \
    --search_config ../search_configs/transformer_softmax_large.json