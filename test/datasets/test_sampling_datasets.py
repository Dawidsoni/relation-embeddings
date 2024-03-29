from unittest.mock import MagicMock

import tensorflow as tf
import gin.tf
import numpy as np

from datasets.dataset_utils import DatasetType
from datasets.sampling_datasets import SamplingEdgeDataset, SamplingNeighboursDataset
from layers.embeddings_layers import ObjectType
from optimization.loss_objects import NormLossObject


class TestSamplingDatasets(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.clear_config()
        gin.parse_config("""
            LossObject.regularization_strength = 0.0
        """)
        self.edge_object_types = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]

    def test_sampling_dataset(self):
        np.random.seed(2)
        dataset = SamplingEdgeDataset(
            dataset_id="dataset1", dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH,
            shuffle_dataset=False, negatives_per_positive=2, batch_size=4, inference_mode=False,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        pos_samples1, array_of_neg_samples1 = batch1
        pos_samples2, array_of_neg_samples2 = batch2
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1["object_ids"])
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples2["object_ids"])
        self.assertAllEqual([[0, 0, 0], [0, 1, 2], [2, 0, 1], [1, 1, 0]],  array_of_neg_samples1[0]["object_ids"])
        self.assertAllEqual([[0, 0, 2], [2, 1, 2], [1, 0, 1], [1, 1, 1]],  array_of_neg_samples1[1]["object_ids"])
        self.assertAllEqual([[2, 0, 1], [1, 1, 0], [0, 0, 2], [2, 1, 2]],  array_of_neg_samples2[0]["object_ids"])
        self.assertAllEqual([[1, 0, 1], [1, 1, 1], [0, 0, 0], [0, 1, 2]],  array_of_neg_samples2[1]["object_ids"])
        expected_object_types = tf.broadcast_to(self.edge_object_types, shape=(4, 3))
        self.assertAllEqual(expected_object_types,  pos_samples1["object_types"])
        self.assertAllEqual(expected_object_types,  pos_samples2["object_types"])
        self.assertAllEqual(expected_object_types,  array_of_neg_samples1[0]["object_types"])
        self.assertAllEqual(expected_object_types,  array_of_neg_samples1[1]["object_types"])
        self.assertAllEqual(expected_object_types,  array_of_neg_samples2[0]["object_types"])
        self.assertAllEqual(expected_object_types,  array_of_neg_samples2[1]["object_types"])
        self.assertAllEqual([False, True, True, False],  array_of_neg_samples1[0]["head_swapped"])
        self.assertAllEqual([False, True, True, False],  array_of_neg_samples1[1]["head_swapped"])
        self.assertAllEqual([True, False, False, True],  array_of_neg_samples2[0]["head_swapped"])
        self.assertAllEqual([True, False, False, True],  array_of_neg_samples2[1]["head_swapped"])

    def test_sampling_dataset_sample_weights_model(self):
        sample_weights_model_mock = MagicMock(return_value=tf.random.uniform(shape=(4, 3)))
        dataset = SamplingEdgeDataset(
            dataset_id="dataset1", dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH,
            shuffle_dataset=False, negatives_per_positive=1, batch_size=4, sample_weights_count=2,
            sample_weights_model=sample_weights_model_mock, inference_mode=False,
            sample_weights_loss_object=NormLossObject(order=1, margin=3.0),
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        pos_samples1, array_of_neg_samples1 = batch1
        pos_samples2, array_of_neg_samples2 = batch2
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1["object_ids"])
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 3)), pos_samples1["object_types"])
        self.assertAllEqual([[0, 0, 1], [1, 1, 2], [0, 0, 1], [1, 1, 2]], pos_samples1["object_ids"])
        self.assertAllEqual(tf.broadcast_to(self.edge_object_types, shape=(4, 3)),  pos_samples2["object_types"])

    def test_sampling_neighbours_dataset_two_neighbours(self):
        np.random.seed(2)
        dataset = SamplingNeighboursDataset(
            dataset_id="dataset1", dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH,
            shuffle_dataset=False, negatives_per_positive=1, batch_size=3, neighbours_per_sample=2,
            inference_mode=False,
        )
        samples_batch = next(iter(dataset.samples))
        positive_samples, negative_samples = samples_batch[0], samples_batch[1][0]
        self.assertAllEqual(
            np.array([[2, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2],
                      [0, 1, 2, 1, 0, 1, 2, 1, 1, 1, 2],
                      [0, 0, 2, 1, 0, 1, 2, 1, 1, 1, 2]]),
            positive_samples["object_ids"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
                      [0, 1, 0, 0, 1, 2, 2, 0, 1, 2, 2],
                      [0, 1, 0, 0, 1, 2, 2, 0, 1, 2, 2]]),
            positive_samples["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6],
                      [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6],
                      [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]]),
            positive_samples["positions"]
        )
        self.assertAllEqual(
            np.array([[2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                      [2, 0, 2, 1, 2, 1, 2, 1, 1, 1, 2]]),
            negative_samples["object_ids"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2],
                      [0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
                      [0, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2]]),
            negative_samples["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6],
                      [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6],
                      [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]]),
            negative_samples["positions"]
        )

    def test_sampling_neighbours_dataset_one_neighbour(self):
        np.random.seed(2)
        dataset = SamplingNeighboursDataset(
            dataset_id="dataset1", dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH,
            shuffle_dataset=False, negatives_per_positive=1, batch_size=3, neighbours_per_sample=1,
            inference_mode=False,
        )
        samples_batch = next(iter(dataset.samples))
        positive_samples, negative_samples = samples_batch[0], samples_batch[1][0]
        self.assertAllEqual(
            np.array([[2, 0, 0, 1, 2, 1, 2], [0, 1, 2, 1, 0, 1, 1], [0, 0, 2, 1, 0, 1, 1]]),
            positive_samples["object_ids"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 0, 2, 2, 2, 2], [0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 0, 1]]),
            positive_samples["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]),
            positive_samples["positions"]
        )
        self.assertAllEqual(
            np.array([[2, 0, 1, 1, 2, 0, 0], [1, 1, 2, 1, 2, 1, 2], [2, 0, 2, 1, 2, 1, 1]]),
            negative_samples["object_ids"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 0, 2, 2, 0, 1], [0, 1, 0, 2, 2, 2, 2], [0, 1, 0, 2, 2, 0, 1]]),
            negative_samples["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]),
            negative_samples["positions"]
        )
