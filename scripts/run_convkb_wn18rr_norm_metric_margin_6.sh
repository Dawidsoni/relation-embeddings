#!/usr/bin/env bash

python3 ../src/train_model.py \
    --gin_configs ../configs/wn18rr_convkb_training_config.gin \
    --gin_bindings "LossObject.optimized_metric = %OptimizedMetric.NORM" \
    "LossObject.norm_metric_margin = 6.0" \
    "convkb/ModelConfig.include_reduce_dim_layer = False" \
    "ExperimentConfig.experiment_name = 'convkb_wn18rr_norm_metric_margin_6'"
