import os
from dataclasses import dataclass
from enum import Enum
import functools

import tensorflow as tf
import numpy as np
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from models.convkb_model import ConvKBModel
from models.knowledge_completion_model import KnowledgeCompletionModel
from models.s_transe_model import STranseModel
from models.transe_model import TranseModel
from models.transformer_transe_model import TransformerTranseModel
from optimization.datasets import Dataset, SamplingDataset, DatasetType
from optimization.loss_objects import LossObject, NormLossObject, SoftplusLossObject
from optimization.model_evaluators import ModelEvaluator, SamplingModelEvaluator
from optimization.model_trainers import ModelTrainer, SamplingModelTrainer


@dataclass
class KnowledgeBaseState(object):
    model: KnowledgeCompletionModel
    training_dataset: Dataset
    loss_object: LossObject
    model_trainer: ModelTrainer
    training_evaluator: ModelEvaluator
    validation_evaluator: ModelEvaluator
    test_evaluator: ModelEvaluator


@gin.constants_from_enum
class LossType(Enum):
    NORM = 1
    SOFTPLUS = 2


@gin.constants_from_enum
class ModelType(Enum):
    TRANSE = 1
    STRANSE = 2
    CONVKB = 3
    TRANSFORMER_TRANSE = 4


def _create_loss_object(loss_type: LossType):
    type_mappings = {
        LossType.NORM: lambda: NormLossObject(),
        LossType.SOFTPLUS: lambda: SoftplusLossObject(),
    }
    if loss_type not in type_mappings:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return type_mappings[loss_type]()


def _create_model(embeddings_config: EmbeddingsConfig, model_type: ModelType):
    type_mappings = {
        ModelType.TRANSE: lambda: TranseModel(embeddings_config),
        ModelType.STRANSE: lambda: STranseModel(embeddings_config),
        ModelType.CONVKB: lambda: ConvKBModel(embeddings_config),
        ModelType.TRANSFORMER_TRANSE: lambda: TransformerTranseModel(embeddings_config)
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_dataset(dataset_type, model_type: ModelType, batch_size=None, repeat_samples=False, shuffle_dataset=False):
    sampling_dataset_initializer = functools.partial(
        SamplingDataset,
        dataset_type=dataset_type,
        data_directory=gin.REQUIRED,
        batch_size=batch_size,
        repeat_samples=repeat_samples,
        shuffle_dataset=shuffle_dataset,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_dataset_initializer(),
        ModelType.STRANSE: lambda: sampling_dataset_initializer(),
        ModelType.CONVKB: lambda: sampling_dataset_initializer(),
        ModelType.TRANSFORMER_TRANSE: lambda: sampling_dataset_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _get_existing_graph_edges(model_type: ModelType):
    training_dataset = _create_dataset(DatasetType.TRAINING, model_type)
    validation_dataset = _create_dataset(DatasetType.VALIDATION, model_type)
    test_dataset = _create_dataset(DatasetType.TEST, model_type)
    return training_dataset.graph_edges + validation_dataset.graph_edges + test_dataset.graph_edges


def _create_model_trainer(model_type, model, loss_object, learning_rate_schedule):
    sampling_trainer_initializer = functools.partial(
        SamplingModelTrainer,
        model=model,
        loss_object=loss_object,
        learning_rate_schedule=learning_rate_schedule,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_trainer_initializer(),
        ModelType.STRANSE: lambda: sampling_trainer_initializer(),
        ModelType.CONVKB: lambda: sampling_trainer_initializer(),
        ModelType.TRANSFORMER_TRANSE: lambda: sampling_trainer_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_model_evaluator(outputs_folder, dataset_type, model_type, model, loss_object, learning_rate_scheduler):
    existing_graph_edges = _get_existing_graph_edges(model_type)
    unbatched_dataset = _create_dataset(
        dataset_type, batch_size=None, repeat_samples=False, shuffle_dataset=True, model_type=model_type
    )
    sampling_evaluator_initializer = functools.partial(
        SamplingModelEvaluator,
        model=model,
        loss_object=loss_object,
        dataset=unbatched_dataset,
        existing_graph_edges=existing_graph_edges,
        output_directory=outputs_folder,
        learning_rate_scheduler=learning_rate_scheduler,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.STRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.CONVKB: lambda: sampling_evaluator_initializer(),
        ModelType.TRANSFORMER_TRANSE: lambda: sampling_evaluator_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


@gin.configurable
def _create_learning_rate_schedule(
    initial_learning_rate=gin.REQUIRED, decay_steps=gin.REQUIRED, decay_rate=gin.REQUIRED, staircase=False
):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=staircase
    )


@gin.configurable(blacklist=["model_type"])
def _create_embeddings_config(
    model_type: ModelType,
    entity_embeddings_path=gin.REQUIRED,
    relation_embeddings_path=gin.REQUIRED,
    position_embeddings_path=gin.REQUIRED,
    mask_embeddings_path=gin.REQUIRED,
):
    dataset = _create_dataset(DatasetType.TRAINING, model_type)
    pretrained_entity_embeddings = (
        np.load(entity_embeddings_path) if entity_embeddings_path is not None else None
    )
    pretrained_relation_embeddings = (
        np.load(relation_embeddings_path) if relation_embeddings_path is not None else None
    )
    pretrained_position_embeddings = (
        np.load(position_embeddings_path) if position_embeddings_path is not None else None
    )
    pretrained_mask_embeddings = (
        np.load(mask_embeddings_path) if mask_embeddings_path is not None else None
    )
    return EmbeddingsConfig(
        entities_count=(max(dataset.ids_of_entities) + 1),
        relations_count=(max(dataset.ids_of_relations) + 1),
        pretrained_entity_embeddings=pretrained_entity_embeddings,
        pretrained_relation_embeddings=pretrained_relation_embeddings,
        pretrained_position_embeddings=pretrained_position_embeddings,
        pretrained_mask_embeddings=pretrained_mask_embeddings,
    )


@gin.configurable(blacklist=['tensorboard_folder'])
def create_knowledge_base_state(
    tensorboard_folder: str,
    model_type: ModelType = gin.REQUIRED,
    loss_type: LossType = gin.REQUIRED,
):
    embeddings_config = _create_embeddings_config(model_type)
    learning_rate_scheduler = _create_learning_rate_schedule()
    loss_object = _create_loss_object(loss_type)
    model = _create_model(embeddings_config, model_type)
    training_dataset = _create_dataset(
        DatasetType.TRAINING, batch_size=gin.REQUIRED, repeat_samples=True, shuffle_dataset=True, model_type=model_type
    )
    init_eval = functools.partial(
        _create_model_evaluator,
        model_type=model_type, model=model, loss_object=loss_object, learning_rate_scheduler=learning_rate_scheduler,
    )
    path_func = functools.partial(os.path.join, tensorboard_folder)
    return KnowledgeBaseState(
        model=model,
        training_dataset=training_dataset,
        loss_object=loss_object,
        model_trainer=_create_model_trainer(model_type, model, loss_object, learning_rate_scheduler),
        training_evaluator=init_eval(outputs_folder=path_func("train"), dataset_type=DatasetType.TRAINING),
        validation_evaluator=init_eval(outputs_folder=path_func("validation"), dataset_type=DatasetType.VALIDATION),
        test_evaluator=init_eval(outputs_folder=path_func("test"), dataset_type=DatasetType.TEST),
    )
