import tensorflow as tf
import numpy as np

from datasets import dataset_utils
from datasets.dataset_utils import DatasetType
from datasets.raw_dataset import RawDataset


class TestDatasetUtils(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def test_incremental_edges(self):
        np.random.seed(2)
        dataset = RawDataset(
            dataset_id="dataset1", dataset_type=DatasetType.VALIDATION, data_directory=self.DATASET_PATH,
            shuffle_dataset=False, batch_size=1, inference_mode=False,
        )
        self.assertDictEqual(
            dataset.known_entity_output_edges, {0: [(1, 0)], 1: [(2, 1)]}
        )
        self.assertDictEqual(
            dataset.known_entity_input_edges,  {1: [(0, 0)], 2: [(1, 1)]}
        )

    def test_get_existing_graph_edges(self):
        self.assertAllEqual(
            [(0, 0, 1), (1, 1, 2), (2, 0, 0), (0, 1, 2), (0, 0, 2), (2, 1, 0)],
            dataset_utils.get_existing_graph_edges(self.DATASET_PATH),
        )
