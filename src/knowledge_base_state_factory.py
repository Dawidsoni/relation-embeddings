from typing import Optional
import os
from dataclasses import dataclass
from enum import Enum
import functools

from models.transformer_softmax_model import TransformerSoftmaxModel
from optimization import datasets

import numpy as np
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from models.conve_model import ConvEModel
from models.convkb_model import ConvKBModel
from models.knowledge_completion_model import KnowledgeCompletionModel
from models.s_transe_model import STranseModel
from models.transe_model import TranseModel
from models.transformer_binary_model import TransformerBinaryModel
from models.transformer_transe_model import TransformerTranseModel
from optimization.datasets import Dataset, SamplingEdgeDataset, DatasetType, SamplingNeighboursDataset, \
    MaskedEntityOfEdgeDataset
from optimization.learning_rate_schedulers import PiecewiseLinearDecayScheduler
from optimization.loss_objects import LossObject, NormLossObject, SoftplusLossObject, BinaryCrossEntropyLossObject, \
    CrossEntropyLossObject
from optimization.model_evaluators import ModelEvaluator, SamplingModelEvaluator, SoftmaxModelEvaluator
from optimization.model_trainers import ModelTrainer, SamplingModelTrainer, SoftmaxModelTrainer


@dataclass
class KnowledgeBaseState(object):
    model: KnowledgeCompletionModel
    best_model: KnowledgeCompletionModel
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
    BINARY_CROSS_ENTROPY = 3
    CROSS_ENTROPY = 4


@gin.constants_from_enum
class ModelType(Enum):
    TRANSE = 1
    STRANSE = 2
    CONVKB = 3
    TRANSFORMER_TRANSE = 4
    TRANSFORMER_BINARY = 5
    CONVE = 6
    TRANSFORMER_SOFTMAX = 7


def _create_loss_object(loss_type: LossType):
    type_mappings = {
        LossType.NORM: lambda: NormLossObject(),
        LossType.SOFTPLUS: lambda: SoftplusLossObject(),
        LossType.BINARY_CROSS_ENTROPY: lambda: BinaryCrossEntropyLossObject(),
        LossType.CROSS_ENTROPY: lambda: CrossEntropyLossObject(),
    }
    if loss_type not in type_mappings:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return type_mappings[loss_type]()


def _create_model(embeddings_config: EmbeddingsConfig, model_type: ModelType):
    type_mappings = {
        ModelType.TRANSE: lambda: TranseModel(embeddings_config),
        ModelType.STRANSE: lambda: STranseModel(embeddings_config),
        ModelType.CONVKB: lambda: ConvKBModel(embeddings_config),
        ModelType.TRANSFORMER_TRANSE: lambda: TransformerTranseModel(embeddings_config),
        ModelType.TRANSFORMER_BINARY: lambda: TransformerBinaryModel(embeddings_config),
        ModelType.CONVE: lambda: ConvEModel(embeddings_config),
        ModelType.TRANSFORMER_SOFTMAX: lambda: TransformerSoftmaxModel(embeddings_config),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_dataset(
    dataset_type,
    model_type: ModelType,
    batch_size,
    prefetched_samples,
    shuffle_dataset=False,
    sample_weights_model=None,
    sample_weights_loss_object=None,
):
    sampling_edge_dataset_initializer = functools.partial(
        SamplingEdgeDataset,
        dataset_type=dataset_type,
        data_directory=gin.REQUIRED,
        batch_size=batch_size,
        shuffle_dataset=shuffle_dataset,
        sample_weights_model=sample_weights_model,
        sample_weights_loss_object=sample_weights_loss_object,
        prefetched_samples=prefetched_samples,
    )
    sampling_neighbours_dataset_initializer = functools.partial(
        SamplingNeighboursDataset,
        dataset_type=dataset_type,
        data_directory=gin.REQUIRED,
        batch_size=batch_size,
        neighbours_per_sample=gin.REQUIRED,
        shuffle_dataset=shuffle_dataset,
        sample_weights_model=sample_weights_model,
        sample_weights_loss_object=sample_weights_loss_object,
        prefetched_samples=prefetched_samples,
    )
    masked_entity_of_edge_dataset_initializer = functools.partial(
        MaskedEntityOfEdgeDataset,
        dataset_type=dataset_type,
        data_directory=gin.REQUIRED,
        batch_size=batch_size,
        shuffle_dataset=shuffle_dataset,
        prefetched_samples=prefetched_samples,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_edge_dataset_initializer(),
        ModelType.STRANSE: lambda: sampling_edge_dataset_initializer(),
        ModelType.CONVKB: lambda: sampling_edge_dataset_initializer(),
        ModelType.TRANSFORMER_TRANSE: lambda: sampling_neighbours_dataset_initializer(),
        ModelType.TRANSFORMER_BINARY: lambda: sampling_neighbours_dataset_initializer(),
        ModelType.CONVE: lambda: masked_entity_of_edge_dataset_initializer(),
        ModelType.TRANSFORMER_SOFTMAX: lambda: masked_entity_of_edge_dataset_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_model_trainer(model_type, model, loss_object, learning_rate_schedule):
    sampling_trainer_initializer = functools.partial(
        SamplingModelTrainer,
        model=model,
        loss_object=loss_object,
        learning_rate_schedule=learning_rate_schedule,
    )
    softmax_trainer_initializer = functools.partial(
        SoftmaxModelTrainer,
        model=model,
        loss_object=loss_object,
        learning_rate_schedule=learning_rate_schedule,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_trainer_initializer(),
        ModelType.STRANSE: lambda: sampling_trainer_initializer(),
        ModelType.CONVKB: lambda: sampling_trainer_initializer(),
        ModelType.TRANSFORMER_TRANSE: lambda: sampling_trainer_initializer(),
        ModelType.TRANSFORMER_BINARY: lambda: sampling_trainer_initializer(),
        ModelType.CONVE: lambda: softmax_trainer_initializer(),
        ModelType.TRANSFORMER_SOFTMAX: lambda: softmax_trainer_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_model_evaluator(
    outputs_folder, dataset_type, prefetched_samples, model_type, model, loss_object, learning_rate_scheduler
):
    evaluation_dataset = _create_dataset(
        dataset_type, batch_size=200, shuffle_dataset=True, model_type=model_type,
        prefetched_samples=prefetched_samples,
    )
    existing_graph_edges = datasets.get_existing_graph_edges(evaluation_dataset.data_directory)
    sampling_evaluator_initializer = functools.partial(
        SamplingModelEvaluator,
        model=model,
        loss_object=loss_object,
        dataset=evaluation_dataset,
        existing_graph_edges=existing_graph_edges,
        output_directory=outputs_folder,
        learning_rate_scheduler=learning_rate_scheduler,
    )
    softmax_evaluator_initializer = functools.partial(
        SoftmaxModelEvaluator,
        model=model,
        loss_object=loss_object,
        dataset=evaluation_dataset,
        existing_graph_edges=existing_graph_edges,
        output_directory=outputs_folder,
        learning_rate_scheduler=learning_rate_scheduler,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.STRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.CONVKB: lambda: sampling_evaluator_initializer(),
        ModelType.TRANSFORMER_TRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.TRANSFORMER_BINARY: lambda: sampling_evaluator_initializer(),
        ModelType.CONVE: lambda: softmax_evaluator_initializer(),
        ModelType.TRANSFORMER_SOFTMAX: lambda: softmax_evaluator_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


@gin.configurable(blacklist=["model_type"])
def _create_embeddings_config(
    model_type: ModelType,
    entity_embeddings_path=gin.REQUIRED,
    relation_embeddings_path=gin.REQUIRED,
    position_embeddings_path=gin.REQUIRED,
    special_token_embeddings_path=gin.REQUIRED,
):
    dataset = _create_dataset(DatasetType.TRAINING, model_type, batch_size=1, prefetched_samples=10)
    pretrained_entity_embeddings = (
        np.load(entity_embeddings_path) if entity_embeddings_path is not None else None
    )
    pretrained_relation_embeddings = (
        np.load(relation_embeddings_path) if relation_embeddings_path is not None else None
    )
    pretrained_position_embeddings = (
        np.load(position_embeddings_path) if position_embeddings_path is not None else None
    )
    pretrained_special_token_embeddings = (
        np.load(special_token_embeddings_path) if special_token_embeddings_path is not None else None
    )
    return EmbeddingsConfig(
        entities_count=(max(dataset.ids_of_entities) + 1),
        relations_count=(max(dataset.ids_of_relations) + 1),
        pretrained_entity_embeddings=pretrained_entity_embeddings,
        pretrained_relation_embeddings=pretrained_relation_embeddings,
        pretrained_position_embeddings=pretrained_position_embeddings,
        pretrained_special_token_embeddings=pretrained_special_token_embeddings,
    )


@gin.configurable
def create_sampling_weights_model_with_loss_object(model_type: Optional[ModelType] = None):
    if model_type is None:
        return None, None
    with gin.config_scope("sampling_weights"):
        embeddings_config = _create_embeddings_config(model_type)
        model = _create_model(embeddings_config, model_type=model_type)
        loss_object = _create_loss_object(loss_type=LossType.NORM)
        return model, loss_object


@gin.configurable(blacklist=['tensorboard_folder'])
def create_knowledge_base_state(
    tensorboard_folder: str,
    model_type: ModelType = gin.REQUIRED,
    loss_type: LossType = gin.REQUIRED,
):
    embeddings_config = _create_embeddings_config(model_type)
    learning_rate_scheduler = PiecewiseLinearDecayScheduler()
    loss_object = _create_loss_object(loss_type)
    model = _create_model(embeddings_config, model_type)
    best_model = _create_model(embeddings_config, model_type)
    sample_weights_model, sample_weights_loss_object = create_sampling_weights_model_with_loss_object()
    training_dataset = _create_dataset(
        DatasetType.TRAINING,
        batch_size=gin.REQUIRED,
        shuffle_dataset=True,
        model_type=model_type,
        sample_weights_model=sample_weights_model,
        sample_weights_loss_object=sample_weights_loss_object,
        prefetched_samples=10,
    )
    init_eval = functools.partial(
        _create_model_evaluator,
        model_type=model_type, model=model, loss_object=loss_object, learning_rate_scheduler=learning_rate_scheduler,
    )
    path_func = functools.partial(os.path.join, tensorboard_folder)
    return KnowledgeBaseState(
        model=model,
        best_model=best_model,
        training_dataset=training_dataset,
        loss_object=loss_object,
        model_trainer=_create_model_trainer(model_type, model, loss_object, learning_rate_scheduler),
        training_evaluator=init_eval(
            model=model, outputs_folder=path_func("train"), dataset_type=DatasetType.TRAINING, prefetched_samples=3,
        ),
        validation_evaluator=init_eval(
            model=model, outputs_folder=path_func("validation"), dataset_type=DatasetType.VALIDATION,
            prefetched_samples=3,
        ),
        test_evaluator=init_eval(
            model=best_model, outputs_folder=path_func("test"), dataset_type=DatasetType.TEST, prefetched_samples=1,
        ),
    )
