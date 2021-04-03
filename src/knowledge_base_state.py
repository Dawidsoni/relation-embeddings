from dataclasses import dataclass

from models.knowledge_completion_model import KnowledgeCompletionModel
from optimization.datasets import Dataset
from optimization.loss_objects import LossObject
from optimization.model_evaluators import ModelEvaluator
from optimization.model_trainers import ModelTrainer


@dataclass
class KnowledgeBaseState(object):
    model: KnowledgeCompletionModel
    training_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    loss_object: LossObject
    model_evaluator: ModelEvaluator
    model_trainer: ModelTrainer
