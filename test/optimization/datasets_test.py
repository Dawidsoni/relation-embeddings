import tensorflow as tf
import numpy as np

from optimization.datasets import DatasetType, SamplingDataset, MaskedDataset, ObjectType


class TestDatasets(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        self.edge_object_types = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]

    def test_sampling_dataset_samples(self):
        dataset = SamplingDataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH)
        pairs_of_samples = list(iter(dataset.samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([[0, 0, 1], self.edge_object_types], positive_samples[0])
        self.assertAllEqual([[1, 1, 2], self.edge_object_types], positive_samples[1])
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0]))
        self.assertEqual(0, negative_samples[0][0][1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[1][0]))
        self.assertEqual(1, negative_samples[1][0][1])

    def test_sampling_dataset_pairs_of_samples(self):
        dataset = SamplingDataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH)
        pairs_of_samples = list(iter(dataset.samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([[0, 0, 1], self.edge_object_types], positive_samples[0])
        self.assertAllEqual([[1, 1, 2], self.edge_object_types], positive_samples[1])
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0]))
        self.assertEqual(0, negative_samples[0][0][1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[1][0]))
        self.assertEqual(1, negative_samples[1][0][1])

    def test_sampling_dataset_batch_size(self):
        dataset = SamplingDataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=2)
        pairs_of_samples = list(iter(dataset.samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([0, 0, 1], positive_samples[0][0][0])
        self.assertAllEqual([1, 1, 2], positive_samples[0][0][1])
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0][0]))
        self.assertEqual(0, negative_samples[0][0][0, 1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[0][0][1]))
        self.assertEqual(1, negative_samples[0][0][1, 1])
