import tensorflow as tf
import numpy as np

from optimization.datasets import Dataset, DatasetType
from optimization.edges_producer import EdgesProducer


class TestEdgesProducer(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def test_produce_head_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=None)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges_object_ids, edges_object_types = edges_producer.produce_head_edges(
            sample=(np.array([1, 0, 1]), np.array([1, 0, 2])), target_pattern_index=0
        )
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], edges_object_ids)
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges_object_types)

    def test_produce_tail_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=None)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges_object_ids, edges_object_types = edges_producer.produce_tail_edges(
            sample=(np.array([1, 1, 1]), np.array([1, 0, 2])), target_pattern_index=0
        )
        self.assertAllEqual([[1, 1, 1], [1, 1, 0]], edges_object_ids)
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges_object_types)

    def test_target_pattern_index(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=None)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges_object_ids, edges_object_types = edges_producer.produce_head_edges(
            sample=(np.array([1, 0, 1]), np.array([1, 0, 2])), target_pattern_index=1
        )
        self.assertAllEqual([[2, 0, 1], [1, 0, 1]], edges_object_ids)
        self.assertAllEqual([[1, 0, 2], [1, 0, 2]], edges_object_types)

    def test_edge_pattern_in_existing_edges(self):
        dataset = Dataset(dataset_type=DatasetType.TRAINING, data_directory=self.DATASET_PATH, batch_size=None)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        edges_object_ids, unused_edges_object_types = edges_producer.produce_head_edges(
            sample=(np.array([0, 0, 1]), np.array([1, 0, 2])), target_pattern_index=0
        )
        self.assertAllEqual((3, 3), edges_object_ids.shape)
        self.assertAllEqual([0, 0, 1], edges_object_ids[0])


if __name__ == '__main__':
    tf.test.main()
