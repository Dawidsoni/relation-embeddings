import tensorflow as tf
import numpy as np
from unittest import mock
from optimization.existing_edges_filter import ExistingEdgesFilter


class TestExistingEdgesFilter(tf.test.TestCase):

    def setUp(self):
        self.graph_edges = [(0, 0, 1), (0, 0, 2), (1, 1, 0), (2, 1, 0), (2, 1, 1)]

    @mock.patch.object(np.random, 'randint')
    def test_three_output_edges(self, randint_mock):
        randint_mock.return_value = 0
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=False
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[1, 1, 0], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([1.0, 2.0, 0.5], filtered_values)
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_two_output_edges(self, randint_mock):
        randint_mock.return_value = 0
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=False
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[0, 0, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([2.0, 1.0], filtered_values)
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_one_output_edge(self, randint_mock):
        randint_mock.return_value = 0
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=False
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[2, 1, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([2.0], filtered_values)
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_mask_index(self, randint_mock):
        randint_mock.return_value = 0
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=False
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[1, 1, 0], mask_index=0, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([0.5, 1.0], filtered_values)
        self.assertEqual(0, target_index)

    @mock.patch.object(np.random, 'randint')
    def test_target_index(self, randint_mock):
        randint_mock.return_value = 1
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=False
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[0, 0, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([1.0, 2.0], filtered_values)
        self.assertEqual(1, target_index)

    def test_use_entities_order_three_output_edges(self):
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=True
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[1, 1, 0], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([1.0, 0.5, 2.0], filtered_values)
        self.assertEqual(0, target_index)

    def test_use_entities_order_two_output_edges(self):
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=True
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[0, 0, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([1.0, 2.0], filtered_values)
        self.assertEqual(1, target_index)

    def test_use_entities_order_one_output_edge(self):
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=True
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[2, 1, 2], mask_index=2, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([2.0], filtered_values)
        self.assertEqual(0, target_index)

    def test_use_entities_order_mask_index(self):
        existing_edges_filter = ExistingEdgesFilter(
            entities_count=3, graph_edges=self.graph_edges, use_entities_order=True
        )
        filtered_values, target_index = existing_edges_filter.get_values_corresponding_to_existing_edges(
            edge_ids=[1, 1, 0], mask_index=0, values=np.array([1.0, 0.5, 2.0])
        )
        self.assertAllEqual([1.0, 0.5], filtered_values)
        self.assertEqual(1, target_index)


if __name__ == '__main__':
    tf.test.main()
