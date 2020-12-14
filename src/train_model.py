import argparse
import logging
import os
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
LOGS_LOCATION = "../logs/"
TRAINING_DATASET_FILENAME = "train.txt"
VALIDATION_DATASET_FILENAME = "valid.txt"
TEST_DATASET_FILENAME = "test.txt"
TENSORBOARD_OUTPUTS_FOLDER = "tensorboard"


@gin.configurable
@dataclass
class ExperimentConfig:
    experiment_name: str = gin.REQUIRED
    epochs: int = gin.REQUIRED


@gin.constants_from_enum
class ModelType(Enum):
    TRANSE = 1
    CONVKB = 2


def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_configs', type=str, required=True, nargs='+')
    parser.add_argument('--gin_bindings', type=str, default=[], nargs='+')
    return parser.parse_args()


def init_and_get_logger(experiment_name):
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if not os.path.exists(LOGS_LOCATION):
        os.makedirs(LOGS_LOCATION)
    file_handler = logging.FileHandler(os.path.join(LOGS_LOCATION, f"{experiment_name}.log"), mode='w')
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


def create_training_evaluator(experiment_name, model, loss_object, existing_graph_edges):
    unbatched_training_dataset = Dataset(
        graph_edges_filename=TRAINING_DATASET_FILENAME, batch_size=None, repeat_samples=True
    )
    output_directory = f"../{TENSORBOARD_OUTPUTS_FOLDER}/{experiment_name}/train"
    return ModelEvaluator(model, loss_object, unbatched_training_dataset, existing_graph_edges, output_directory)


def create_validation_evaluator(experiment_name, model, loss_object, existing_graph_edges):
    unbatched_validation_dataset = Dataset(
        graph_edges_filename=VALIDATION_DATASET_FILENAME, batch_size=None, repeat_samples=True
    )
    output_directory = f"../{TENSORBOARD_OUTPUTS_FOLDER}/{experiment_name}/validation"
    return ModelEvaluator(model, loss_object, unbatched_validation_dataset, existing_graph_edges, output_directory)


def train_model(gin_configs, gin_bindings):
    gin.parse_config_files_and_bindings(gin_configs, gin_bindings)
    experiment_config = ExperimentConfig()
    experiment_name, epochs = experiment_config.experiment_name, experiment_config.epochs
    logger = init_and_get_logger(experiment_name)
    log_experiment_information(logger, experiment_name, gin_configs, gin_bindings)
    batched_training_dataset = Dataset(graph_edges_filename=TRAINING_DATASET_FILENAME, batch_size=gin.REQUIRED)
    model = create_model(batched_training_dataset)
    loss_object = LossObject()
    learning_rate_schedule = create_learning_rate_schedule()
    trainer = ModelTrainer(model, loss_object, learning_rate_schedule)
    existing_graph_edges = get_existing_graph_edges()
    training_evaluator = create_training_evaluator(experiment_name, model, loss_object, existing_graph_edges)
    validation_evaluator = create_validation_evaluator(experiment_name, model, loss_object, existing_graph_edges)
    training_step = 0
    for epoch in range(epochs):
        logger.info(f"Training a model (epoch {epoch})")
        for positive_inputs, negative_inputs in batched_training_dataset.pairs_of_samples:
            trainer.train_step(positive_inputs, negative_inputs)
            training_step += 1
        logger.info(f"Evaluating a on training data (epoch {epoch})")
        training_evaluator.evaluation_step(training_step)
        logger.info(f"Evaluating a on validation data (epoch {epoch})")
        validation_evaluator.evaluation_step(training_step)


if __name__ == '__main__':
    training_args = parse_training_args()
    train_model(training_args.gin_configs, training_args.gin_bindings)
