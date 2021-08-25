from typing import Tuple, List, Type
import numpy as np
import gin.tf
from dataclasses import dataclass

from datasets.raw_dataset import RawDataset


@gin.configurable
@dataclass
class PhaseDatasetTemplate(object):
    dataset_id: str
    dataset_template: Type[RawDataset]


@gin.configurable
@dataclass
class TrainingPhaseTemplate(object):
    dataset_templates_probs: List[Tuple[PhaseDatasetTemplate, float]]
    steps: int


@dataclass
class TrainingPhase(object):
    datasets_probs: List[Tuple[RawDataset, float]]
    steps: int


class TrainingDataset(object):

    def __init__(self, training_phases: List[TrainingPhase], logger):
        self.training_phases = training_phases
        self.logger = logger
        self.current_phase_index = 0
        self.current_phase_step = 0
        self.dataset_iterators = None
        self.dataset_probs = None
        self._update_phase_based_properties(phase_index=0)

    def _update_phase_based_properties(self, phase_index):
        self.logger.info(f"Entering training phase {phase_index}")
        if phase_index < 0 or phase_index >= len(self.training_phases):
            self.dataset_iterators = None
            self.dataset_probs = None
            return
        training_phase = self.training_phases[phase_index]
        self.dataset_iterators = [next(iter(dataset.samples)) for dataset, _ in training_phase.datasets_probs]
        self.dataset_probs = np.array([prob for _, prob in training_phase.datasets_probs])
        
    @property
    def samples(self):
        while self.current_phase_index < len(self.training_phases):
            phase = self.training_phases[self.current_phase_index]
            dataset_index = np.random.choice(len(phase.datasets_probs), p=self.dataset_probs)
            yield self.dataset_iterators[dataset_index]
            self.current_phase_step += 1
            if self.current_phase_step >= phase.steps:
                self.current_phase_index += 1
                self.current_phase_step = 0
                self._update_phase_based_properties(self.current_phase_index)
