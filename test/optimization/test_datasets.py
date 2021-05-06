import tensorflow as tf
import numpy as np

from optimization.datasets import DatasetType, SamplingDataset, MaskedEntityDataset
from layers.embeddings_layers import ObjectType


class TestDatasets(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        self.edge_object_types = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]

    def test_sampling_dataset_samples(self):
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False
        )
        pairs_of_samples = list(iter(dataset.samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([[0, 0, 1], self.edge_object_types], positive_samples[0])
        self.assertAllEqual([[1, 1, 2], self.edge_object_types], positive_samples[1])
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0]))
        self.assertEqual(0, negative_samples[0][0][1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[1][0]))
        self.assertEqual(1, negative_samples[1][0][1])

    def test_sampling_dataset_pairs_of_samples(self):
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False
        )
        pairs_of_samples = list(iter(dataset.samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([[0, 0, 1], self.edge_object_types], positive_samples[0])
        self.assertAllEqual([[1, 1, 2], self.edge_object_types], positive_samples[1])
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0]))
        self.assertEqual(0, negative_samples[0][0][1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[1][0]))
        self.assertEqual(1, negative_samples[1][0][1])

    def test_sampling_dataset_batch_size(self):
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=2, shuffle_dataset=False
        )
        pairs_of_samples = list(iter(dataset.samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([0, 0, 1], positive_samples[0][0][0])
        self.assertAllEqual([1, 1, 2], positive_samples[0][0][1])
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0][0]))
        self.assertEqual(0, negative_samples[0][0][0, 1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[0][0][1]))
        self.assertEqual(1, negative_samples[0][0][1, 1])

    def test_masked_dataset_samples(self):
        dataset = MaskedEntityDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False
        )
        samples = list(iter(dataset.samples))
        inputs, outputs, ids_of_outputs, mask_indexes = list(zip(*samples))
        objects_ids, objects_types = list(zip(*inputs))
        self.assertAllEqual([0, 0, 1], objects_ids[0])
        self.assertAllEqual([0, 1, 2], objects_ids[1])
        self.assertAllEqual([0, 0, 0], objects_ids[2])
        self.assertAllEqual([1, 1, 0], objects_ids[3])
        self.assertAllEqual([2, 1, 0], objects_types[0])
        self.assertAllEqual([2, 1, 0], objects_types[1])
        self.assertAllEqual([0, 1, 2], objects_types[2])
        self.assertAllEqual([0, 1, 2], objects_types[3])
        self.assertAllEqual([1.0, 0.0, 0.0], outputs[0])
        self.assertAllEqual([0.0, 1.0, 0.0], outputs[1])
        self.assertAllEqual([0.0, 1.0, 0.0], outputs[2])
        self.assertAllEqual([0.0, 0.0, 1.0], outputs[3])
        self.assertAllEqual([0, 1, 1, 2], ids_of_outputs)
        self.assertAllEqual([0, 0, 2, 2], mask_indexes)

    def test_masked_dataset_batch_size(self):
        dataset = MaskedEntityDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2
        )
        samples = list(iter(dataset.samples))
        inputs, outputs, ids_of_outputs, mask_indexes = list(zip(*samples))
        objects_ids, objects_types = list(zip(*inputs))
        self.assertAllEqual([[0, 0, 1], [0, 1, 2]], objects_ids[0])
        self.assertAllEqual([[0, 0, 0], [1, 1, 0]], objects_ids[1])
        self.assertAllEqual([[2, 1, 0], [2, 1, 0]], objects_types[0])
        self.assertAllEqual([[0, 1, 2], [0, 1, 2]], objects_types[1])
        self.assertAllEqual([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], outputs[0])
        self.assertAllEqual([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], outputs[1])
        self.assertAllEqual([[0, 1], [1, 2]], ids_of_outputs)
        self.assertAllEqual([[0, 0], [2, 2]], mask_indexes)
