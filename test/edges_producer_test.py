import tensorflow as tf
import numpy as np

from dataset import Dataset
from edges_producer import EdgesProducer


class TestEdgesCandidatesProducer(tf.test.TestCase):
    DATASET_PATH = '../test_data'

    def test_produce_head_edges(self):
        dataset = Dataset(graph_edges_filename='graph_edges.txt', data_directory=self.DATASET_PATH)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        head_edges = edges_producer.produce_head_edges(edge_pattern=np.array([1, 0, 1]), target_pattern_index=0)
        self.assertAllEqual([[1, 0, 1], [2, 0, 1]], head_edges)

    def test_produce_tail_edges(self):
        dataset = Dataset(graph_edges_filename='graph_edges.txt', data_directory=self.DATASET_PATH)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        tail_edges = edges_producer.produce_tail_edges(edge_pattern=np.array([1, 1, 1]), target_pattern_index=0)
        self.assertAllEqual([[1, 1, 1], [1, 1, 0]], tail_edges)

    def test_target_pattern_index(self):
        dataset = Dataset(graph_edges_filename='graph_edges.txt', data_directory=self.DATASET_PATH)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        head_edges = edges_producer.produce_head_edges(edge_pattern=np.array([1, 0, 1]), target_pattern_index=1)
        self.assertAllEqual([[2, 0, 1], [1, 0, 1]], head_edges)

    def test_edge_pattern_in_existing_edges(self):
        dataset = Dataset(graph_edges_filename='graph_edges.txt', data_directory=self.DATASET_PATH)
        edges_producer = EdgesProducer(dataset.ids_of_entities, dataset.graph_edges)
        head_edges = edges_producer.produce_head_edges(edge_pattern=np.array([0, 0, 1]), target_pattern_index=0)
        self.assertAllEqual((3, 3), head_edges.shape)
        self.assertAllEqual([0, 0, 1], head_edges[0])


if __name__ == '__main__':
    tf.test.main()
