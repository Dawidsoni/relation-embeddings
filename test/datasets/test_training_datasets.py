import tensorflow as tf
import numpy as np
import unittest.mock

from datasets.dataset_utils import DatasetType
from datasets.softmax_datasets import MaskedEntityOfEdgeDataset, MaskedEntityOfPathDataset
from datasets.training_datasets import TrainingPhase, TrainingDataset


class TestDatasetUtils(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        self.batch1_dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=1,
        )
        self.batch2_dataset = MaskedEntityOfPathDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_samples=10
        )
        self.batch3_dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=3
        )

    def test_deterministic_phases(self):
        phases = [
            TrainingPhase(datasets_probs=[(self.batch2_dataset, 1.0)], steps=3),
            TrainingPhase(datasets_probs=[(self.batch3_dataset, 1.0)], steps=2),
            TrainingPhase(datasets_probs=[(self.batch2_dataset, 1.0)], steps=1),
        ]
        training_dataset = TrainingDataset(phases, logger=unittest.mock.MagicMock())
        batch_sizes = [x["object_ids"].shape[0] for x in training_dataset.samples]
        self.assertAllEqual([2, 2, 2, 3, 3, 2], batch_sizes)

    def test_non_deterministic_phases(self):
        np.random.seed(1)
        phases = [
            TrainingPhase(datasets_probs=[(self.batch3_dataset, 0.3), (self.batch2_dataset, 0.7)], steps=3),
            TrainingPhase(datasets_probs=[(self.batch3_dataset, 0.6), (self.batch2_dataset, 0.4)], steps=2),
        ]
        training_dataset = TrainingDataset(phases, logger=unittest.mock.MagicMock())
        batch_sizes = [x["object_ids"].shape[0] for x in training_dataset.samples]
        self.assertAllEqual([2, 2, 3, 3, 3], batch_sizes)

    def test_different_samples_yielded(self):
        phases = [
            TrainingPhase(datasets_probs=[(self.batch1_dataset, 1.0)], steps=2),
        ]
        training_dataset = TrainingDataset(phases, logger=unittest.mock.MagicMock())
        samples = list(training_dataset.samples)
        self.assertNotAllEqual(samples[0]["object_ids"], samples[1]["object_ids"])

    def test_different_phases_samples_equal(self):
        phases = [
            TrainingPhase(datasets_probs=[(self.batch2_dataset, 1.0)], steps=1),
            TrainingPhase(datasets_probs=[(self.batch2_dataset, 1.0)], steps=1),
        ]
        training_dataset = TrainingDataset(phases, logger=unittest.mock.MagicMock())
        samples = list(training_dataset.samples)
        for key in samples[0].keys():
            self.assertAllEqual(samples[0][key], samples[1][key])
