import tensorflow as tf
import numpy as np

from datasets.dataset_utils import DatasetType
from datasets.softmax_datasets import MaskedEntityOfEdgeDataset, MaskedEntityOfPathDataset, MaskedRelationOfEdgeDataset, \
    InputNeighboursDataset, OutputNeighboursDataset, InputOutputNeighboursDataset


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

    def test_input_neighbours_dataset_training_1_neighbour(self):
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1, 2, 1], [0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 0, 1], [2, 1, 0, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0, 1, 2], [1, 1, 0, 0, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[0, 1, 2, 2, 2], [0, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_neighbours_dataset_training_2_neighbours(self):
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=2, mask_source_entity_pbty=0.0, inference_mode=False,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 2, 1, 1, 2], [0, 1, 2, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [2, 1, 0, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2, 1, 2], [1, 1, 0, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_neighbours_dataset_training_mask_source_entity(self):
        np.random.seed(10)
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=False,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 3, 2, 1], [0, 1, 3, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 0, 1], [2, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[3, 0, 0, 1, 2], [3, 1, 0, 0, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 2, 2], [2, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"])

    def test_input_neighbours_dataset_validation_inference(self):
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=True,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [0, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0, 1, 0], [0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 0, 1], [2, 1, 0, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 2], [2, 0, 0]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 2, 1, 2], [2, 0, 0, 1, 1]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [0, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["expected_output"])

    def test_output_neighbours_dataset_training_1_neighbour(self):
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1, 1, 2], [0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [2, 1, 0, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0, 1, 2], [1, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[0, 1, 2, 2, 2], [0, 1, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_output_neighbours_dataset_training_mask_source_entity(self):
        np.random.seed(10)
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=False,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 3, 1, 2], [0, 1, 3, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 2, 2], [2, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[3, 0, 0, 1, 2], [3, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 2, 2], [2, 1, 2, 2, 2]], dtype=np.int32), batch2["object_types"])

    def test_output_neighbours_dataset_validation_inference_1_neighbour(self):
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=True,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [0, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0, 1, 2], [0, 1, 2, 1, 1]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [2, 1, 0, 0, 1]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 2], [2, 0, 0]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 2, 1, 1], [2, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 0, 1], [0, 1, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["expected_output"])

    def test_output_neighbours_dataset_validation_inference_2_neighbours(self):
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=2, mask_source_entity_pbty=1.0, inference_mode=True,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [0, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2, 1, 2], [0, 1, 2, 1, 1, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [2, 1, 0, 0, 1, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 2], [2, 0, 0]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 2, 1, 1, 1, 2], [2, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_training_1_neighbour(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, mask_input_context_pbty=0.0,
            mask_output_context_pbty=0.0,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 2, 1, 1, 2], [0, 1, 2, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [2, 1, 0, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2, 1, 2], [1, 1, 0, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_training_mask_source_entity(self):
        np.random.seed(10)
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=False, mask_input_context_pbty=0.0,
            mask_output_context_pbty=0.0,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(
            np.array([[0, 0, 3, 2, 1, 1, 2], [0, 1, 3, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 2, 0, 1, 2, 2], [2, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[3, 0, 0, 1, 2, 1, 2], [3, 1, 0, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 2, 2, 2, 2, 2], [2, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])

    def test_input_output_neighbours_dataset_validation_inference_1_neighbour(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=True, mask_input_context_pbty=1.0,
            mask_output_context_pbty=1.0,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [0, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 0, 1, 2], [0, 1, 2, 1, 2, 1, 1]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [2, 1, 0, 2, 2, 0, 1]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 2], [2, 0, 0]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 2, 1, 2, 1, 1], [2, 0, 0, 1, 1, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 0, 1], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_validation_inference_2_neighbours(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=2, mask_source_entity_pbty=1.0, inference_mode=True, mask_input_context_pbty=1.0,
            mask_output_context_pbty=1.0,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [0, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 0, 1, 2, 1, 2, 1, 2], [0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2]], dtype=np.int32),
            batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2], [2, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2]], dtype=np.int32),
            batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 2], [2, 0, 0]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 2, 1, 2, 1, 2, 1, 1, 1, 2], [2, 0, 0, 1, 1, 1, 2, 1, 2, 1, 2]], dtype=np.int32),
            batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2], [0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2]], dtype=np.int32),
            batch2["object_types"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_mask_input_context(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, mask_input_context_pbty=1.0,
            mask_output_context_pbty=0.0,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 4, 5, 1, 2], [0, 1, 2, 4, 5, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [2, 1, 0, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 4, 5, 1, 2], [1, 1, 0, 4, 5, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_mask_output_context(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, mask_input_context_pbty=0.0,
            mask_output_context_pbty=1.0,
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 2, 1, 4, 5], [0, 1, 2, 1, 2, 4, 5]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [2, 1, 0, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2, 4, 5], [1, 1, 0, 0, 0, 4, 5]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([2, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([0, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])
