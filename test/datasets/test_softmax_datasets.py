import tensorflow as tf
import gin.tf
import numpy as np

from datasets.dataset_utils import DatasetType
from datasets.softmax_datasets import MaskedEntityOfEdgeDataset, MaskedEntityOfPathDataset, MaskedRelationOfEdgeDataset, \
    InputNeighboursDataset, OutputNeighboursDataset, InputOutputNeighboursDataset, CombinedMaskedDataset, \
    CombinedMaskedDatasetTrainingMode


class TestSoftmaxDatasets(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        gin.enter_interactive_mode()

    def test_masked_entity_of_edge_dataset(self):
        dataset = MaskedEntityOfEdgeDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            inference_mode=False, dataset_id="dataset1"
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
            max_samples=10, inference_mode=False, dataset_id="dataset1"
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
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            inference_mode=False, dataset_id="dataset1",
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
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1, 2, 1], [0, 0, 0, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 0, 1], [0, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2, 1, 2], [1, 1, 0, 0, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [0, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_neighbours_dataset_training_2_neighbours(self):
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=2, mask_source_entity_pbty=0.0, inference_mode=False, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 2, 1, 1, 2], [0, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 3, 4, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 2, 1, 2], [1, 1, 0, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 3, 4, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_neighbours_dataset_training_mask_source_entity(self):
        np.random.seed(10)
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=False, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 3, 2, 1], [3, 0, 0, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 0, 1], [2, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 1, 3, 1, 2], [3, 1, 0, 0, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 2, 2], [2, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"])

    def test_input_neighbours_dataset_validation_inference(self):
        dataset = InputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=True, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0, 1, 0], [2, 0, 0, 1, 1]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 0, 1], [0, 1, 2, 0, 1]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2, 1, 2], [0, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [0, 1, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_output_neighbours_dataset_training_1_neighbour(self):
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 1, 1, 2], [0, 0, 0, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [0, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2, 1, 2], [1, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [0, 1, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_output_neighbours_dataset_training_mask_source_entity(self):
        np.random.seed(10)
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=False, dataset_id="dataset1"
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 3, 1, 2], [3, 0, 0, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 2, 2], [2, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 1, 3, 1, 2], [3, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 2, 2, 2], [2, 1, 2, 2, 2]], dtype=np.int32), batch2["object_types"])

    def test_output_neighbours_dataset_validation_inference_1_neighbour(self):
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=True, dataset_id="dataset1"
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(np.array([[0, 0, 0, 1, 2], [2, 0, 0, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 2, 2], [0, 1, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(np.array([[0, 1, 2, 1, 1], [0, 1, 0, 1, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(np.array([[2, 1, 0, 0, 1], [0, 1, 2, 0, 1]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_output_neighbours_dataset_validation_inference_2_neighbours(self):
        dataset = OutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=2, mask_source_entity_pbty=1.0, inference_mode=True, dataset_id="dataset1"
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 2, 1, 2], [2, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 3, 4, 3, 4]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 1, 1, 2], [0, 1, 0, 1, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4], [0, 1, 2, 3, 4, 3, 4]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_training_1_neighbour(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, mask_input_context_pbty=0.0,
            mask_output_context_pbty=0.0, dataset_id="dataset1"
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 2, 1, 1, 2], [0, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 2, 1, 2], [1, 1, 0, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_training_mask_source_entity(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=False, mask_input_context_pbty=0.0,
            mask_output_context_pbty=0.0, dataset_id="dataset1"
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(
            np.array([[0, 0, 3, 2, 1, 1, 2], [3, 0, 0, 1, 2, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 2, 0, 1, 2, 2], [2, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 3, 1, 2, 1, 2], [3, 1, 0, 0, 0, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 2, 2, 2, 2, 2], [2, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])

    def test_input_output_neighbours_dataset_validation_inference_1_neighbour(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=1.0, inference_mode=True, mask_input_context_pbty=1.0,
            mask_output_context_pbty=1.0, dataset_id="dataset1"
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 0, 1, 2], [2, 0, 0, 1, 1, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 2, 1, 1], [0, 1, 0, 1, 2, 1, 0]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 0, 1], [0, 1, 2, 2, 2, 0, 1]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_validation_inference_2_neighbours(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=2, mask_source_entity_pbty=1.0, inference_mode=True, mask_input_context_pbty=1.0,
            mask_output_context_pbty=1.0, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 0, 1, 0, 1, 2, 1, 2, 1, 2], [2, 0, 0, 1, 1, 1, 2, 1, 2, 1, 2]], dtype=np.int32),
            batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2]], dtype=np.int32),
            batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6], [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]], dtype=np.int32),
            batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2], [0, 1, 0, 1, 2, 1, 2, 1, 0, 1, 2]], dtype=np.int32),
            batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2], [0, 1, 2, 2, 2, 2, 2, 0, 1, 2, 2]], dtype=np.int32),
            batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6], [0, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]], dtype=np.int32),
            batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_mask_input_context(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, mask_input_context_pbty=1.0,
            mask_output_context_pbty=0.0, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 4, 5, 1, 2], [0, 0, 0, 4, 5, 1, 2]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 4, 5, 1, 2], [1, 1, 0, 4, 5, 1, 2]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_input_output_neighbours_dataset_mask_output_context(self):
        dataset = InputOutputNeighboursDataset(
            dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, shuffle_dataset=False, batch_size=2,
            max_neighbours_count=1, mask_source_entity_pbty=0.0, inference_mode=False, mask_input_context_pbty=0.0,
            mask_output_context_pbty=1.0, dataset_id="dataset1",
        )
        samples_iterator = iter(dataset.samples)
        batch1, batch2 = next(samples_iterator), next(samples_iterator)
        self.assertAllEqual(np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32), batch1["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 0, 1, 2, 1, 4, 5], [0, 0, 0, 1, 2, 4, 5]], dtype=np.int32), batch1["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 0, 1, 2, 2], [0, 1, 2, 2, 2, 2, 2]], dtype=np.int32), batch1["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch1["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch1["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch1["true_entity_index"])
        self.assertAllEqual(np.array([0, 1], dtype=np.int32), batch1["expected_output"])
        self.assertAllEqual(np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32), batch2["edge_ids"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 1, 2, 4, 5], [1, 1, 0, 0, 0, 4, 5]], dtype=np.int32), batch2["object_ids"])
        self.assertAllEqual(
            np.array([[2, 1, 0, 2, 2, 2, 2], [0, 1, 2, 0, 1, 2, 2]], dtype=np.int32), batch2["object_types"])
        self.assertAllEqual(
            np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.int32), batch2["positions"])
        self.assertAllEqual(np.array([0, 2], dtype=np.int32), batch2["mask_index"])
        self.assertAllEqual(np.array([2, 0], dtype=np.int32), batch2["true_entity_index"])
        self.assertAllEqual(np.array([1, 2], dtype=np.int32), batch2["expected_output"])

    def test_combined_masked_datasets(self):
        gin.parse_config("""
            InputNeighboursDataset.dataset_id = "input_dataset"
            InputNeighboursDataset.max_neighbours_count = 1
            InputNeighboursDataset.mask_source_entity_pbty = 0.5
            OutputNeighboursDataset.dataset_id = "output_dataset"
            OutputNeighboursDataset.max_neighbours_count = 2
            OutputNeighboursDataset.mask_source_entity_pbty = 0.25
        """)
        template1 = InputNeighboursDataset
        template2 = OutputNeighboursDataset
        dataset = CombinedMaskedDataset(
            dataset_templates=[template1, template2], dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH, shuffle_dataset=True, batch_size=4, inference_mode=False,
            dataset_id="combined_dataset", training_mode=CombinedMaskedDatasetTrainingMode.INDEPENDENT_LOSSES,
        )
        samples = next(iter(dataset.samples))
        self.assertAllEqual(samples["edge_ids"], samples["input_dataset@edge_ids"])
        self.assertAllEqual(samples["edge_ids"], samples["output_dataset@edge_ids"])
        self.assertAllEqual(samples["mask_index"], samples["input_dataset@mask_index"])
        self.assertAllEqual(samples["mask_index"], samples["output_dataset@mask_index"])
        self.assertAllEqual(samples["true_entity_index"], samples["input_dataset@true_entity_index"])
        self.assertAllEqual(samples["true_entity_index"], samples["output_dataset@true_entity_index"])
        self.assertAllEqual(samples["expected_output"], samples["input_dataset@expected_output"])
        self.assertAllEqual(samples["expected_output"], samples["output_dataset@expected_output"])
        self.assertAllEqual((4, 5), samples["input_dataset@object_ids"].shape)
        self.assertAllEqual((4, 7), samples["output_dataset@object_ids"].shape)
        self.assertAllEqual(
            samples["mode"],
            4 * [CombinedMaskedDatasetTrainingMode.INDEPENDENT_LOSSES.value]
        )

    def test_combined_masked_datasets_inference_mode(self):
        gin.parse_config("""
            InputNeighboursDataset.dataset_id = "input_dataset"
            InputNeighboursDataset.max_neighbours_count = 1
            InputNeighboursDataset.mask_source_entity_pbty = 0.5
            OutputNeighboursDataset.dataset_id = "output_dataset"
            OutputNeighboursDataset.max_neighbours_count = 2
            OutputNeighboursDataset.mask_source_entity_pbty = 0.25
        """)
        template1 = InputNeighboursDataset
        template2 = OutputNeighboursDataset
        dataset = CombinedMaskedDataset(
            dataset_templates=[template1, template2], dataset_type=DatasetType.TRAINING,
            data_directory=self.DATASET_PATH, shuffle_dataset=True, batch_size=4, inference_mode=True,
            dataset_id="combined_dataset", training_mode=CombinedMaskedDatasetTrainingMode.INDEPENDENT_LOSSES,
        )
        samples = next(iter(dataset.samples))
        self.assertAllEqual(
            samples["mode"],
            4 * [CombinedMaskedDatasetTrainingMode.JOINT_LOSS.value]
        )
