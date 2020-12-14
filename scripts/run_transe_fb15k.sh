#!/usr/bin/env bash

python3 ../src/train_model.py \
  --gin_configs ../configs/training_config.gin \
  --gin_bindings "Dataset.data_directory='../data/FB15k-237'"
