import tensorflow as tf
import numpy as np

from dataset import Dataset


class TestDatasets(tf.test.TestCase):
    DATASET_PATH = '../test_data'

    def test_samples(self):
        dataset = Dataset(self.DATASET_PATH, graph_edges_filename='graph_edges.txt')
        positive_samples = list(iter(dataset.positive_samples))
        self.assertAllEqual([0, 0, 1], positive_samples[0])
        self.assertAllEqual([1, 1, 2], positive_samples[1])
        negative_samples = list(iter(dataset.negative_samples))
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0]))
        self.assertEqual(0, negative_samples[0][1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[1]))
        self.assertEqual(1, negative_samples[1][1])

    def test_pairs_of_samples(self):
        dataset = Dataset(self.DATASET_PATH, graph_edges_filename='graph_edges.txt')
        pairs_of_samples = list(iter(dataset.pairs_of_samples))
        positive_samples, negative_samples = list(zip(*pairs_of_samples))
        self.assertAllEqual([0, 0, 1], positive_samples[0])
        self.assertAllEqual([1, 1, 2], positive_samples[1])
        negative_samples = list(iter(dataset.negative_samples))
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0]))
        self.assertEqual(0, negative_samples[0][1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[1]))
        self.assertEqual(1, negative_samples[1][1])

    def test_batch_size(self):
        dataset = Dataset(self.DATASET_PATH, graph_edges_filename='graph_edges.txt', batch_size=2)
        positive_samples = list(iter(dataset.positive_samples))
        self.assertAllEqual([0, 0, 1], positive_samples[0][0])
        self.assertAllEqual([1, 1, 2], positive_samples[0][1])
        negative_samples = list(iter(dataset.negative_samples))
        self.assertEqual(1, np.sum([0, 0, 1] != negative_samples[0][0]))
        self.assertEqual(0, negative_samples[0][0, 1])
        self.assertEqual(1, np.sum([1, 1, 2] != negative_samples[0][1]))
        self.assertEqual(1, negative_samples[0][1, 1])

    def test_repeat_samples(self):
        dataset = Dataset(self.DATASET_PATH, graph_edges_filename='graph_edges.txt', repeat_samples=True)
        positive_iterator = iter(dataset.positive_samples)
        positive_samples = [next(positive_iterator), next(positive_iterator), next(positive_iterator)]
        self.assertAllEqual(positive_samples[0], positive_samples[2])
        self.assertGreater(np.sum(positive_samples[0] != positive_samples[1]), 0)


if __name__ == '__main__':
    tf.test.main()
