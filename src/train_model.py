import argparse
import logging
import os
import time
import tensorflow as tf
import numpy as np
import gin.tf
from dataclasses import dataclass
from enum import Enum

from dataset import Dataset
from edges_producer import EdgesProducer
from losses import LossObject
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from conv_base_model import DataConfig
from transe_model import TranseModel
from s_transe_model import STranseModel
from convkb_model import ConvKBModel
from evaluation_metrics import EvaluationMetrics


DEFAULT_LOGGER_NAME = "default_logger"
TRAINING_DATASET_FILENAME = "train.txt"
VALIDATION_DATASET_FILENAME = "valid.txt"
TEST_DATASET_FILENAME = "test.txt"


@gin.configurable
@dataclass
class ExperimentConfig:
    experiment_name: str = gin.REQUIRED
    training_steps: int = gin.REQUIRED
    steps_per_evaluation: int = gin.REQUIRED
    tensorboard_outputs_folder: str = gin.REQUIRED
    model_save_folder: str = gin.REQUIRED
    logs_output_folder: str = gin.REQUIRED


@gin.constants_from_enum
class ModelType(Enum):
    TRANSE = 1
    STRANSE = 2
    CONVKB = 3


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
    pretrained_entity_embeddings = (
        np.load(entity_embeddings_path) if entity_embeddings_path is not None else None
    )
    pretrained_relations_embeddings = (
        np.load(relations_embeddings_path) if relations_embeddings_path is not None else None
    )
    entities_count = max(training_dataset.ids_of_entities) + 1
    relations_count = max(training_dataset.ids_of_relations) + 1
    data_config = DataConfig(
        entities_count, relations_count, pretrained_entity_embeddings, pretrained_relations_embeddings
    )
    if model_type == ModelType.TRANSE:
        return TranseModel(data_config)
    elif model_type == ModelType.STRANSE:
        return STranseModel(data_config)
    elif model_type == ModelType.CONVKB:
        return ConvKBModel(data_config)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


@gin.configurable
def create_learning_rate_schedule(initial_learning_rate, decay_steps, decay_rate, staircase=False):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=staircase
    )


def get_existing_graph_edges():
    training_dataset = Dataset(graph_edges_filename=TRAINING_DATASET_FILENAME)
    validation_dataset = Dataset(graph_edges_filename=VALIDATION_DATASET_FILENAME)
    test_dataset = Dataset(graph_edges_filename=TEST_DATASET_FILENAME)
    return training_dataset.graph_edges + validation_dataset.graph_edges + test_dataset.graph_edges


def create_training_evaluator(tensorboard_folder, model, loss_object, learning_rate_scheduler=None):
    existing_graph_edges = get_existing_graph_edges()
    unbatched_training_dataset = Dataset(
        graph_edges_filename=TRAINING_DATASET_FILENAME, batch_size=None, repeat_samples=True
    )
    outputs_folder = os.path.join(tensorboard_folder, "train")
    return ModelEvaluator(
        model, loss_object, unbatched_training_dataset, existing_graph_edges, outputs_folder, learning_rate_scheduler
    )


def create_validation_evaluator(tensorboard_folder, model, loss_object):
    existing_graph_edges = get_existing_graph_edges()
    unbatched_validation_dataset = Dataset(
        graph_edges_filename=VALIDATION_DATASET_FILENAME, batch_size=None, repeat_samples=True
    )
    output_directory = os.path.join(tensorboard_folder, "validation")
    return ModelEvaluator(model, loss_object, unbatched_validation_dataset, existing_graph_edges, output_directory)


def evaluate_and_log_test_metrics(model, loss_object, logger):
    test_dataset = Dataset(graph_edges_filename=TEST_DATASET_FILENAME, batch_size=None, repeat_samples=False)
    existing_graph_edges = get_existing_graph_edges()
    edges_producer = EdgesProducer(test_dataset.ids_of_entities, existing_graph_edges)
    samples_iterator = test_dataset.positive_samples.as_numpy_iterator()
    named_metrics = EvaluationMetrics.compute_metrics_on_samples(
        model, loss_object, edges_producer, samples_iterator
    )
    for name_prefix, metrics in named_metrics.items():
        mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
        logger.info(f"Evaluating a model on test dataset: {name_prefix}/mean_rank: {mean_rank}")
        logger.info(f"Evaluating a model on test dataset: {name_prefix}/mean_reciprocal_rank: {mean_reciprocal_rank}")
        logger.info(f"Evaluating a model on test dataset: {name_prefix}/hits10: {hits10}")


def train_and_evaluate_model(experiment_config, experiment_id, logger):
    batched_training_dataset = Dataset(
        graph_edges_filename=TRAINING_DATASET_FILENAME, batch_size=gin.REQUIRED, repeat_samples=True
    )
    model = create_model(batched_training_dataset)
    loss_object = LossObject()
    learning_rate_scheduler = create_learning_rate_schedule()
    trainer = ModelTrainer(model, loss_object, learning_rate_scheduler)
    tensorboard_folder = os.path.join(experiment_config.tensorboard_outputs_folder, experiment_id)
    training_evaluator = create_training_evaluator(tensorboard_folder, model, loss_object, learning_rate_scheduler)
    validation_evaluator = create_validation_evaluator(tensorboard_folder, model, loss_object)
    training_samples = batched_training_dataset.pairs_of_samples.take(experiment_config.training_steps)
    training_step = 1
    for positive_inputs, negative_inputs in training_samples:
        logger.info(f"Performing training step {training_step}")
        trainer.train_step(positive_inputs, negative_inputs, training_step)
        if training_step % experiment_config.steps_per_evaluation == 0:
            logger.info(f"Evaluating a model on training data")
            training_evaluator.evaluation_step(training_step)
            logger.info(f"Evaluating a model on validation data")
            validation_evaluator.evaluation_step(training_step)
        training_step += 1
    evaluate_and_log_test_metrics(model, loss_object, logger)
    save_path_of_model = os.path.join(experiment_config.model_save_folder, experiment_id)
    model.save_with_embeddings(save_path_of_model)
    logger.info(f"Model saved in '{save_path_of_model}'")


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
