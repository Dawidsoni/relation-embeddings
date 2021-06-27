import tensorflow as tf
import numpy as np

from optimization.datasets import Dataset, DatasetType
from optimization.edges_producer import EdgesProducer


class TestEdgesProducer(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def test_produce_head_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        head_edges = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2])}, target_pattern_index=0
        )
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], head_edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], head_edges["object_types"])

    def test_produce_tail_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        tail_edges = edges_producer.produce_tail_edges(
            sample={"object_ids": np.array([1, 1, 1]), "object_types": np.array([1, 0, 2])}, target_pattern_index=0
        )
        self.assertAllEqual([[1, 1, 1], [1, 1, 0]], tail_edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], tail_edges["object_types"])

    def test_target_pattern_index(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2])}, target_pattern_index=1
        )
        self.assertAllEqual([[2, 0, 1], [1, 0, 1]], edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges["object_types"])

    def test_edge_pattern_in_existing_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([0, 0, 1]), "object_types": np.array([1, 0, 2])}, target_pattern_index=0
        )
        self.assertAllEqual((3, 3), edges["object_ids"].shape)
        self.assertAllEqual([0, 0, 1], edges["object_ids"][0])

    def test_multiple_object_ids(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1, 2, 3]), "object_types": np.array([1, 0, 2, 1, 0])},
            target_pattern_index=0
        )
        self.assertAllEqual([[1, 0, 1, 2, 3], [2, 0, 1, 2, 3]], edges["object_ids"])
        self.assertAllEqual([[1, 0, 2, 1, 0], [1, 0, 2, 1, 0]], edges["object_types"])

    def test_edge_information(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=1)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges = edges_producer.produce_head_edges(
            sample={"object_ids": np.array([1, 0, 1]), "object_types": np.array([1, 0, 2]),
                    "position": 5},
            target_pattern_index=0
        )
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], edges["object_ids"])
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges["object_types"])
        self.assertAllEqual([5, 5], edges["position"])


if __name__ == '__main__':
    tf.test.main()
