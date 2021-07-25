import tensorflow as tf
import numpy as np
from unittest import mock

from optimization.datasets import Dataset, DatasetType
from optimization.edges_producer import EdgesProducer


class TestEdgesProducer(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    @mock.patch.object(np.random, 'randint')
    def test_produce_head_edges(self, randint_mock):
        randint_mock.return_value = 0
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        head_edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2])}
        )
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], head_edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], head_edges["object_types"])
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_produce_tail_edges(self, randint_mock):
        randint_mock.return_value = 0
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        tail_edges, target_index = edges_producer.produce_tail_edges(
            sample={"object_ids": np.array([1, 1, 1]), "object_types": np.array([1, 0, 2])}
        )
        self.assertAllEqual([[1, 1, 1], [1, 1, 0]], tail_edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], tail_edges["object_types"])
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_target_pattern_index(self, randint_mock):
        randint_mock.return_value = 1
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2])}
        )
        self.assertAllEqual([[2, 0, 1], [1, 0, 1]], edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges["object_types"])
        self.assertEqual(1, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_edge_pattern_in_existing_edges(self, randint_mock):
        randint_mock.return_value = 0
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([0, 0, 1]), "object_types": np.array([1, 0, 2])}
        )
        self.assertAllEqual((3, 3), edges["object_ids"].shape)
        self.assertAllEqual([0, 0, 1], edges["object_ids"][0])
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_multiple_object_ids(self, randint_mock):
        randint_mock.return_value = 0
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1, 2, 3]), "object_types": np.array([1, 0, 2, 1, 0])},
        )
        self.assertAllEqual([[1, 0, 1, 2, 3], [2, 0, 1, 2, 3]], edges["object_ids"])
        self.assertAllEqual([[1, 0, 2, 1, 0], [1, 0, 2, 1, 0]], edges["object_types"])
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_edge_information(self, randint_mock):
        randint_mock.return_value = 0
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2]),
                    "position": 5},
        )
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges["object_types"])
        self.assertAllEqual([5, 5], edges["position"])
        self.assertEqual(0, target_index)

    def test_use_entities_order_two_output_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges, use_entities_order=True)
        head_edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2])}
        )
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], head_edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], head_edges["object_types"])
        self.assertEqual(0, target_index)

    def test_use_entities_order_three_output_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges, use_entities_order=True)
        head_edges, target_index = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 1, 1]), "object_types": np.array([1, 0, 2])}
        )
        self.assertAllEqual([[0, 1, 1], [1, 1, 1], [2, 1, 1]], head_edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2], [1, 0, 2]], head_edges["object_types"])
        self.assertEqual(1, target_index)


if __name__ == '__main__':
    tf.test.main()
