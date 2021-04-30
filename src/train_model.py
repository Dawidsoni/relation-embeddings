import argparse
import logging
import os
import time
import gin.tf
import numpy as np
import collections
from dataclasses import dataclass

import knowledge_base_state_factory


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


class TrainingStopper(object):

    def __init__(self, losses_to_keep=1500, expected_improvement=0.995):
        self.losses_to_keep = losses_to_keep
        self.losses_queue = collections.deque()
        self.expected_improvement = expected_improvement

    def add_loss_value(self, loss_value):
        self.losses_queue.append(loss_value)
        while len(self.losses_queue) > self.losses_to_keep:
            self.losses_queue.popleft()

    def should_training_stop(self):
        if len(self.losses_queue) < self.losses_to_keep:
            return False
        split_index = int(self.losses_to_keep / 2)
        prev_elements_mean = np.median(list(self.losses_queue)[:split_index])
        last_elements_mean = np.median(list(self.losses_queue)[split_index:])
        return prev_elements_mean == 0.0 or last_elements_mean / prev_elements_mean > self.expected_improvement


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', type=str, required=True, nargs='+')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='+')
    return parser.parse_args()


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


def log_experiment_information(logger, experiment_name, gin_configs, gin_bindings):
    logger.info(f"Starting experiment '{experiment_name}'")
    for gin_config in gin_configs:
        with open(gin_config, mode='r') as file_stream:
            file_content = "".join(file_stream.readlines())
            log_separator = 120 * "="
            logger.info(f"Using Gin configuration: {gin_config}:\n{file_content}\n{log_separator}")
    for gin_binding in gin_bindings:
        logger.info(f"Using Gin binding: {gin_binding}")


def train_and_evaluate_model(experiment_config, experiment_id, logger):
    tensorboard_folder = os.path.join(experiment_config.tensorboard_outputs_folder, experiment_id)
    state = knowledge_base_state_factory.create_knowledge_base_state(tensorboard_folder)
    training_stopper = TrainingStopper()
    training_step = 1
    for training_samples in state.training_dataset.samples.take(experiment_config.training_steps):
        logger.info(f"Performing training step {training_step}")
        loss_value = state.model_trainer.train_step(training_samples, training_step)
        print(f"Loss value: {loss_value: .3f}")
        training_stopper.add_loss_value(loss_value)
        if training_stopper.should_training_stop():
            logger.info(f"Finishing experiment due to high value of computed loss: {loss_value}")
            return
        if training_step % experiment_config.steps_per_evaluation == 0:
            logger.info(f"Evaluating a model on training data")
            state.training_evaluator.evaluation_step(training_step)
            logger.info(f"Evaluating a model on validation data")
            state.validation_evaluator.evaluation_step(training_step)
        if training_step == 1:
            logger.info(f"Parameters count of a model: {state.model.count_params()}")
        training_step += 1
    state.test_evaluator.log_metrics(logger)
    path_to_save_model = os.path.join(experiment_config.model_save_folder, experiment_id)
    state.model.save_with_embeddings(path_to_save_model)


def prepare_and_train_model(gin_configs, gin_bindings):
    gin.parse_config_files_and_bindings(gin_configs, gin_bindings)
    experiment_config = ExperimentConfig()
    experiment_id = f"{experiment_config.experiment_name}_{int(time.time())}"
    logger = init_and_get_logger(experiment_config.logs_output_folder, experiment_id)
    log_experiment_information(logger, experiment_config.experiment_name, gin_configs, gin_bindings)
    train_and_evaluate_model(experiment_config, experiment_id, logger)


if __name__ == '__main__':
    training_args = parse_training_args()
    prepare_and_train_model(training_args.gin_configs, training_args.gin_bindings)
