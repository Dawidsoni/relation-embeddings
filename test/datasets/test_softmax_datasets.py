import tensorflow as tf
import numpy as np

from datasets.dataset_utils import DatasetType
from datasets.softmax_datasets import MaskedEntityOfEdgeDataset, MaskedEntityOfPathDataset, MaskedRelationOfEdgeDataset


class TestSoftmaxDatasets(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def test_masked_entity_of_edge_dataset(self):
        dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 0]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2], [1, 1, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_masked_entity_of_path_dataset(self):
        np.random.seed(10)
        dataset = MaskedEntityOfPathDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_samples=10
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 2, 2], [0, 2, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 2, 0, 1], [0, 0, 0, 1]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 0, 1, 1], [0, 2, 1, 1]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 2, 3, 4], [0, 2, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([1, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 2, 2], [0, 2, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 2, 0, 1], [0, 0, 0, 1]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 0, 1, 1], [0, 2, 1, 1]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([[0, 2, 3, 4], [0, 2, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([1, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_masked_relation_of_edge_dataset(self):
        dataset = MaskedRelationOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[0, 2, 0], [0, 2, 0]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([1, 1], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([-1, -1], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([3, 4], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[0, 2, 0], [0, 2, 0]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([1, 1], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([-1, -1], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([3, 4], dtype=np.int32), batch2["expected_output"])

"""    def test_masked_all_neighbours_dataset_training(self):
        dataset = MaskedAllNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 0]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2], [1, 1, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.int32), batch1["expected_output"])

    def test_masked_all_neighbours_dataset_validation(self):
        np.random.seed(3)
        dataset = MaskedAllNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_masked_entity_with_neighbours_dataset_two_neighbours(self):
        np.random.seed(2)
        dataset = MaskedEntityWithNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            neighbours_per_sample=2,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2, 1, 2], [2, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"]
        )
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"],
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 5, 6, 5, 6]], dtype=np.int32), batch1["positions"],
        )
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 1, 1, 2], [0, 1, 0, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"]
        )
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 5, 6, 5, 6]], dtype=np.int32), batch2["positions"]
        )
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_masked_entity_with_neighbours_dataset_one_neighbour(self):
        np.random.seed(2)
        dataset = MaskedEntityWithNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            neighbours_per_sample=1,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2], [2, 0, 0, 1, 2]], dtype=np.int32), batch1["object_ids"]
        )
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2], [0, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"],
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4], [0, 1, 2, 5, 6]], dtype=np.int32), batch1["positions"],
        )
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 1], [0, 1, 0, 1, 0]], dtype=np.int32), batch2["object_ids"]
        )
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1], [0, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4], [0, 1, 2, 5, 6]], dtype=np.int32), batch2["positions"]
        )
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_reversed_edge_decorator_dataset(self):
        decorated_dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2
        )
        dataset = ReversedEdgeDecoratorDataset(decorated_dataset)
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[1, 2, 0], [0, 0, 0]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[2, 3, 0], [1, 1, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_reversed_edge_decorator_dataset_multiple_object_ids(self):
        np.random.seed(2)
        decorated_dataset = MaskedEntityWithNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            neighbours_per_sample=2,
        )
        dataset = ReversedEdgeDecoratorDataset(decorated_dataset)
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 2, 0, 1, 2, 1, 2], [2, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"],
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 5, 6, 5, 6]], dtype=np.int32), batch1["positions"],
        )
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[2, 3, 0, 1, 1, 1, 2], [0, 1, 0, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 0, 1, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"]
        )
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 5, 6, 5, 6]], dtype=np.int32), batch2["positions"]
        )
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])
"""
