import argparse
import gin.tf
import tensorflow as tf
import logging
import os
from dataclasses import dataclass


DEFAULT_LOGGER_NAME = "default_logger"


@gin.configurable
@dataclass
class ExperimentConfig(object):
    experiment_name: str = gin.REQUIRED
    training_steps: int = gin.REQUIRED
    steps_per_evaluation: int = gin.REQUIRED
    tensorboard_outputs_folder: str = gin.REQUIRED
    model_save_folder: str = gin.REQUIRED
    logs_output_folder: str = gin.REQUIRED


def parse_gin_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', type=str, required=True, nargs='+')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='+')
    return parser.parse_args()


def init_gin_configurables():
    gin.external_configurable(tf.reduce_max, module='tf')
    gin.external_configurable(tf.reduce_min, module='tf')


def init_and_get_logger(logs_location, experiment_id):
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if not os.path.exists(logs_location):
        os.makedirs(logs_location)
    file_handler = logging.FileHandler(os.path.join(logs_location, f"{experiment_id}.log"), mode='w')
    formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.INFO)
    return logger
