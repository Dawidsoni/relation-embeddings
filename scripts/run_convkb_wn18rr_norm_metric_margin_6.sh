#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_convkb_training_config.gin \
    --gin_bindings "create_knowledge_base_state.loss_type = %LossType.NORM" \
    "NormLossObject.margin = 6.0" \
    "convkb/ConvModelConfig.include_reduce_dim_layer = False" \
    "ExperimentConfig.experiment_name = 'convkb_wn18rr_norm_metric_margin_6'"
