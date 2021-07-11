import tensorflow as tf
import numpy as np

from optimization.existing_edges_filter import ExistingEdgesFilter


class TestExistingEdgesFilter(tf.test.TestCase):

    def setUp(self):
        graph_edges = [(0, 0, 1), (0, 0, 2), (1, 1, 0), (2, 1, 0), (2, 1, 1)]
        self.existing_edges_filter = ExistingEdgesFilter(entities_count=3, graph_edges=graph_edges)

    def test_one_output_edge(self):
        filtered_values = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[1, 1, 0], mask_index=2, values=np.array([1.0, 0.5, 2.0]), target_index=0
        )
        self.assertAllEqual([1.0, 2.0, 0.5], filtered_values)

    def test_two_output_edges(self):
        filtered_values = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[0, 0, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0]), target_index=0
        )
        self.assertAllEqual([2.0, 1.0], filtered_values)

    def test_three_edges(self):
        filtered_values = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[2, 1, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0]), target_index=0
        )
        self.assertAllEqual([2.0], filtered_values)

    def test_mask_index(self):
        filtered_values = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[1, 1, 0], mask_index=0, values=np.array([1.0, 0.5, 2.0]), target_index=0
        )
        self.assertAllEqual([0.5, 1.0], filtered_values)

    def test_target_index(self):
        filtered_values = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[0, 0, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0]), target_index=1
        )
        self.assertAllEqual([1.0, 2.0], filtered_values)


if __name__ == '__main__':
    tf.test.main()
