from unittest.mock import MagicMock

import tensorflow as tf
import gin.tf
import numpy as np

from optimization.datasets import DatasetType, MaskedEntityDataset, SamplingDataset
from layers.embeddings_layers import ObjectType
from optimization.loss_objects import NormLossObject


class TestDatasets(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.clear_config()
        gin.parse_config("""
            LossObject.regularization_strength = 0.0
        """)
        self.edge_object_types = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]

    def test_sampling_dataset(self):
        np.random.seed(2)
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH,
            shuffle_dataset=False,
            negatives_per_positive=2,
            batch_size=4,
            repeat_samples=True
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        pos_samples1, array_of_neg_samples1 = batch1
        array_of_neg_ids1, array_of_neg_types1 = array_of_neg_samples1
        pos_samples2, array_of_neg_samples2 = batch2
        array_of_neg_ids2, array_of_neg_types2 = array_of_neg_samples2
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1[0])
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 3)),  pos_samples1[1])
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1[0])
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 3)),  pos_samples2[1])
        self.assertAllEqual(
            [[[0, 0, 0],  [0, 0, 2]], [[0, 1, 2], [2, 1, 2]], [[2, 0, 1], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]],
            array_of_neg_ids1
        )
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 2, 3)), array_of_neg_types1)
        self.assertAllEqual(
            [[[2, 0, 1], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]], [[0, 0, 2], [0, 0, 0]], [[2, 1, 2], [0, 1, 2]]],
            array_of_neg_ids2
        )
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 2, 3)), array_of_neg_types2)

    def test_sampling_dataset_sample_weights_model(self):
        sample_weights_model_mock = MagicMock(return_value=tf.random.uniform(shape=(4, 3)))
        dataset = SamplingDataset(
            dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH,
            shuffle_dataset=False,
            negatives_per_positive=1,
            batch_size=4,
            repeat_samples=True,
            sample_weights_model=sample_weights_model_mock,
            sample_weights_loss_object=NormLossObject(order=1, margin=3.0),
            weights_candidates_count=2
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        pos_samples1, array_of_neg_samples1 = batch1
        pos_samples2, array_of_neg_samples2 = batch2
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1[0])
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 3)),  pos_samples1[1])
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1[0])
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 3)),  pos_samples2[1])

    def test_masked_dataset(self):
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
