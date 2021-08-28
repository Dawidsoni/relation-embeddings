import logging
from typing import List
import os
from dataclasses import dataclass
from enum import Enum
import functools

from datasets.dataset_utils import DatasetType
from datasets.sampling_datasets import SamplingEdgeDataset
from datasets.softmax_datasets import MaskedEntityOfEdgeDataset
from datasets.training_datasets import TrainingDataset, TrainingPhase, TrainingPhaseTemplate, PhaseDatasetTemplate
from models.transformer_softmax_model import TransformerSoftmaxModel

import numpy as np
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from models.conve_model import ConvEModel
from models.convkb_model import ConvKBModel
from models.knowledge_completion_model import KnowledgeCompletionModel
from models.s_transe_model import STranseModel
from models.transe_model import TranseModel
from optimization.learning_rate_schedulers import PiecewiseLinearDecayScheduler
from optimization.loss_objects import LossObject, NormLossObject, SoftplusLossObject, BinaryCrossEntropyLossObject, \
    CrossEntropyLossObject
from optimization.model_evaluators import ModelEvaluator, SamplingModelEvaluator, SoftmaxModelEvaluator
from optimization.model_trainers import ModelTrainer, SamplingModelTrainer, SoftmaxModelTrainer


@dataclass
class KnowledgeBaseState(object):
    model: KnowledgeCompletionModel
    best_model: KnowledgeCompletionModel
    training_dataset: TrainingDataset
    loss_object: LossObject
    model_trainer: ModelTrainer
    training_evaluator: ModelEvaluator
    validation_evaluator: ModelEvaluator
    test_evaluator: ModelEvaluator


class PhaseDatasetTemplateResolver(object):

    def __init__(self):
        self.ids_to_datasets = {}

    def resolve_phase_dataset_template(self, template: PhaseDatasetTemplate):
        if template.dataset_id not in self.ids_to_datasets:
            self.ids_to_datasets[template.dataset_id] = template.dataset_template(
                dataset_type=DatasetType.TRAINING, shuffle_dataset=True, prefetched_samples=10
            )
        return self.ids_to_datasets[template.dataset_id]


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
    CONVE = 4
    TRANSFORMER_SOFTMAX = 5


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
        ModelType.CONVE: lambda: ConvEModel(embeddings_config),
        ModelType.TRANSFORMER_SOFTMAX: lambda: TransformerSoftmaxModel(embeddings_config),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_inference_dataset(
    dataset_type,
    model_type: ModelType,
    batch_size,
    prefetched_samples,
    shuffle_dataset=False,
    sample_weights_model=None,
    sample_weights_loss_object=None,
):
    common_args = {
        "dataset_type": dataset_type, "batch_size": batch_size, "shuffle_dataset": shuffle_dataset,
        "prefetched_samples": prefetched_samples
    }
    sampling_args = {
        "sample_weights_model": sample_weights_model, "sample_weights_loss_object": sample_weights_loss_object,
        "neighbours_per_sample": gin.REQUIRED,
    }
    sampling_edge_dataset_initializer = functools.partial(SamplingEdgeDataset, **common_args, **sampling_args)
    masked_entity_of_edge_dataset_initializer = functools.partial(MaskedEntityOfEdgeDataset, **common_args)
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_edge_dataset_initializer(),
        ModelType.STRANSE: lambda: sampling_edge_dataset_initializer(),
        ModelType.CONVKB: lambda: sampling_edge_dataset_initializer(),
        ModelType.CONVE: lambda: masked_entity_of_edge_dataset_initializer(),
        ModelType.TRANSFORMER_SOFTMAX: lambda: masked_entity_of_edge_dataset_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _resolve_training_template(template):
    return template(dataset_type=DatasetType.TRAINING, shuffle_dataset=True, prefetched_samples=10)


@gin.configurable
def _create_training_dataset(logger, training_phase_templates: List[TrainingPhaseTemplate] = gin.REQUIRED):
    resolver = PhaseDatasetTemplateResolver()
    training_phases = []
    for phase_template in training_phase_templates:
        datasets_probs = [
            (resolver.resolve_phase_dataset_template(dataset_template), probability)
            for dataset_template, probability in phase_template.dataset_templates_probs
        ]
        training_phases.append(TrainingPhase(datasets_probs=datasets_probs, steps=phase_template.steps))
    return TrainingDataset(training_phases, logger)


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
        ModelType.CONVE: lambda: softmax_trainer_initializer(),
        ModelType.TRANSFORMER_SOFTMAX: lambda: softmax_trainer_initializer(),
    }
    if model_type not in type_mappings:
        raise ValueError(f"Invalid model type: {model_type}")
    return type_mappings[model_type]()


def _create_model_evaluator(
    outputs_folder, dataset_type, prefetched_samples, model_type, model, loss_object, learning_rate_scheduler
):
    dataset_initializer = functools.partial(
        _create_inference_dataset, dataset_type=dataset_type, shuffle_dataset=True, model_type=model_type,
        prefetched_samples=prefetched_samples
    )
    sampling_evaluator_initializer = functools.partial(
        SamplingModelEvaluator,
        model=model,
        loss_object=loss_object,
        dataset=dataset_initializer(batch_size=200),
        output_directory=outputs_folder,
        learning_rate_scheduler=learning_rate_scheduler,
    )
    softmax_evaluator_initializer = functools.partial(
        SoftmaxModelEvaluator,
        model=model,
        loss_object=loss_object,
        dataset=dataset_initializer(batch_size=512),
        output_directory=outputs_folder,
        learning_rate_scheduler=learning_rate_scheduler,
    )
    type_mappings = {
        ModelType.TRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.STRANSE: lambda: sampling_evaluator_initializer(),
        ModelType.CONVKB: lambda: sampling_evaluator_initializer(),
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
    dataset = _create_inference_dataset(
        DatasetType.TRAINING, model_type, batch_size=1, prefetched_samples=10
    )
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
        entities_count=dataset.entities_count,
        relations_count=dataset.relations_count,
        pretrained_entity_embeddings=pretrained_entity_embeddings,
        pretrained_relation_embeddings=pretrained_relation_embeddings,
        pretrained_position_embeddings=pretrained_position_embeddings,
        pretrained_special_token_embeddings=pretrained_special_token_embeddings,
    )


@gin.configurable(blacklist=['tensorboard_folder', 'logger'])
def create_knowledge_base_state(
    tensorboard_folder: str,
    logger: logging.Logger,
    model_type: ModelType = gin.REQUIRED,
    loss_type: LossType = gin.REQUIRED,
):
    embeddings_config = _create_embeddings_config(model_type)
    learning_rate_scheduler = PiecewiseLinearDecayScheduler()
    loss_object = _create_loss_object(loss_type)
    model = _create_model(embeddings_config, model_type)
    best_model = _create_model(embeddings_config, model_type)
    training_dataset = _create_training_dataset(logger)
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
