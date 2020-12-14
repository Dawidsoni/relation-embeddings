import argparse
import logging
import os
import time
import tensorflow as tf
import gin.tf
from dataclasses import dataclass
from enum import Enum

from dataset import Dataset
from losses import LossObject
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from conv_base_model import DataConfig
from transe_model import TranseModel
from convkb_model import ConvKBModel


DEFAULT_LOGGER_NAME = "default_logger"
TRAINING_DATASET_FILENAME = "train.txt"
VALIDATION_DATASET_FILENAME = "valid.txt"
TEST_DATASET_FILENAME = "test.txt"


@gin.configurable
@dataclass
class ExperimentConfig:
    experiment_name: str = gin.REQUIRED
    epochs: int = gin.REQUIRED
    steps_per_evaluation: int = gin.REQUIRED
    tensorboard_outputs_folder: str = gin.REQUIRED
    logs_output_folder: str = gin.REQUIRED


@gin.constants_from_enum
class ModelType(Enum):
    TRANSE = 1
    CONVKB = 2


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


@gin.configurable
def create_model(
        training_dataset, model_type=gin.REQUIRED, entity_embeddings_path=gin.REQUIRED,
        relations_embeddings_path=gin.REQUIRED
):
    pretrained_entity_embeddings = None  # TODO: Support loading embeddings
    pretrained_relations_embeddings = None  # TODO: Support loading embeddings
    entities_count = max(training_dataset.ids_of_entities) + 1
    relations_count = max(training_dataset.ids_of_relations) + 1
    data_config = DataConfig(
        entities_count, relations_count, pretrained_entity_embeddings, pretrained_relations_embeddings
    )
    if model_type == ModelType.TRANSE:
        return TranseModel(data_config)
    elif model_type == ModelType.CONVKB:
        return ConvKBModel(data_config)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


@gin.configurable
def create_learning_rate_schedule(initial_learning_rate, decay_steps, decay_rate):
    return tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)


def get_existing_graph_edges():
    training_dataset = Dataset(graph_edges_filename=TRAINING_DATASET_FILENAME)
    validation_dataset = Dataset(graph_edges_filename=VALIDATION_DATASET_FILENAME)
    test_dataset = Dataset(graph_edges_filename=TEST_DATASET_FILENAME)
    return training_dataset.graph_edges + validation_dataset.graph_edges + test_dataset.graph_edges


def create_training_evaluator(tensorboard_folder, model, loss_object, existing_graph_edges):
    unbatched_training_dataset = Dataset(
        graph_edges_filename=TRAINING_DATASET_FILENAME, batch_size=None, repeat_samples=True
    )
    outputs_folder = os.path.join(tensorboard_folder, "train")
    return ModelEvaluator(model, loss_object, unbatched_training_dataset, existing_graph_edges, outputs_folder)


def create_validation_evaluator(tensorboard_folder, model, loss_object, existing_graph_edges):
    unbatched_validation_dataset = Dataset(
        graph_edges_filename=VALIDATION_DATASET_FILENAME, batch_size=None, repeat_samples=True
    )
    output_directory = os.path.join(tensorboard_folder, "validation")
    return ModelEvaluator(model, loss_object, unbatched_validation_dataset, existing_graph_edges, output_directory)


def train_model(gin_configs, gin_bindings):
    gin.parse_config_files_and_bindings(gin_configs, gin_bindings)
    experiment_config = ExperimentConfig()
    experiment_id = f"{experiment_config.experiment_name}_{int(time.time())}"
    logger = init_and_get_logger(experiment_config.logs_output_folder, experiment_id)
    log_experiment_information(logger, experiment_config.experiment_name, gin_configs, gin_bindings)
    batched_training_dataset = Dataset(graph_edges_filename=TRAINING_DATASET_FILENAME, batch_size=gin.REQUIRED)
    model = create_model(batched_training_dataset)
    loss_object = LossObject()
    learning_rate_schedule = create_learning_rate_schedule()
    trainer = ModelTrainer(model, loss_object, learning_rate_schedule)
    existing_graph_edges = get_existing_graph_edges()
    tensorboard_folder = os.path.join(experiment_config.tensorboard_outputs_folder, experiment_id)
    training_evaluator = create_training_evaluator(tensorboard_folder, model, loss_object, existing_graph_edges)
    validation_evaluator = create_validation_evaluator(tensorboard_folder, model, loss_object, existing_graph_edges)
    training_step = 0
    for epoch in range(experiment_config.epochs):
        logger.info(f"Starting epoch {epoch})")
        for positive_inputs, negative_inputs in batched_training_dataset.pairs_of_samples:
            logger.info(f"Performing training step {training_step}")
            trainer.train_step(positive_inputs, negative_inputs)
            training_step += 1
            if training_step % experiment_config.steps_per_evaluation == 0:
                logger.info(f"Evaluating a model on training data")
                training_evaluator.evaluation_step(training_step)
                logger.info(f"Evaluating a model on validation data")
                validation_evaluator.evaluation_step(training_step)


if __name__ == '__main__':
    training_args = parse_training_args()
    train_model(training_args.gin_configs, training_args.gin_bindings)
