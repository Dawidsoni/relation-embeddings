import dataclasses
import os
import pickle
import numpy as np
import tensorflow as tf
from unittest import mock
import builtins

from optimization.similar_entities_producer import SimilarEntitiesProducerConfig, EntitiesSimilarityType, \
    SimilarEntitiesProducer, SimilarEntitiesStorage


class TestSimilarEntitiesProducer(tf.test.TestCase):
    DATASET_PATH = '../../data/test_data'

    def setUp(self):
        self.default_config = SimilarEntitiesProducerConfig(
            cache_path="/tmp",
            dataset_name="test_dataset",
            graph_edges=[(0, 0, 1), (0, 1, 1), (1, 0, 2), (1, 1, 2), (2, 0, 3), (2, 2, 3), (3, 2, 2)],
            entities_count=4,
            relations_count=3,
            similarity_type=EntitiesSimilarityType.OUTPUT_RELATIONS,
            similar_entities_count=2,
        )

    @mock.patch.object(os.path, "exists")
    @mock.patch.object(pickle, "dump")
    def test_create_similar_entities_output_relations_type(self, dump_mock, path_exists_mock):
        path_exists_mock.return_value = False
        similar_entities_producer = SimilarEntitiesProducer(self.default_config)
        self.assertAllEqual(
            [[1, 2], [3, 1], [0, 2], [1, 2]],
            similar_entities_producer.produce_similar_entities(entity_ids=[0, 2, 1, 0])
        )
        self.assertAllClose(
            [[0., 0.666, 1.], [0., 0.666, 1.], [0.5, 0.666, 0.666], [0.5, 1., 1.]],
            similar_entities_producer.similar_entities_storage.similar_entities_distances,
            atol=1e-3,
        )
        path_exists_mock.assert_called_once_with("/tmp/test_dataset_OUTPUT_RELATIONS_max4.pickle")
        dump_mock.assert_called_once_with(similar_entities_producer.similar_entities_storage, mock.ANY)

    @mock.patch.object(os.path, "exists")
    @mock.patch.object(pickle, "dump")
    def test_create_similar_entities_one_entity_fetched(self, dump_mock, path_exists_mock):
        path_exists_mock.return_value = False
        similar_entities_producer = SimilarEntitiesProducer(dataclasses.replace(
            self.default_config, similar_entities_count=1
        ))
        self.assertAllEqual(
            [[1], [3], [0], [1]],
            similar_entities_producer.produce_similar_entities(entity_ids=[0, 2, 1, 0])
        )
        self.assertAllClose(
            [[0., 0.666, 1.], [0., 0.666, 1.], [0.5, 0.666, 0.666], [0.5, 1., 1.]],
            similar_entities_producer.similar_entities_storage.similar_entities_distances,
            atol=1e-3,
        )
        path_exists_mock.assert_called_once_with("/tmp/test_dataset_OUTPUT_RELATIONS_max4.pickle")
        dump_mock.assert_called_once_with(similar_entities_producer.similar_entities_storage, mock.ANY)

    @mock.patch.object(os.path, "exists")
    @mock.patch.object(pickle, "dump")
    def test_create_similar_entities_input_relations_type(self, dump_mock, path_exists_mock):
        path_exists_mock.return_value = False
        similar_entities_producer = SimilarEntitiesProducer(dataclasses.replace(
            self.default_config, similarity_type=EntitiesSimilarityType.INPUT_RELATIONS
        ))
        self.assertAllEqual(
            [[3, 2], [1, 3], [2, 3], [3, 2]],
            similar_entities_producer.produce_similar_entities(entity_ids=[0, 2, 1, 0])
        )
        self.assertAllClose(
            [[1., 1., 1.], [0.333, 0.666, 1.], [0.333, 0.333, 1.], [0.333, 0.666, 1.]],
            similar_entities_producer.similar_entities_storage.similar_entities_distances,
            atol=1e-3,
        )
        path_exists_mock.assert_called_once_with("/tmp/test_dataset_INPUT_RELATIONS_max4.pickle")
        dump_mock.assert_called_once_with(similar_entities_producer.similar_entities_storage, mock.ANY)

    @mock.patch.object(os.path, "exists")
    @mock.patch.object(pickle, "load")
    @mock.patch.object(builtins, "open")
    def test_load_similar_entities_storage(self, open_mock, load_mock, path_exists_mock):
        path_exists_mock.return_value = True
        load_mock.return_value = SimilarEntitiesStorage(
            similar_entities_ids=np.array([[1], [0]], dtype=np.int32),
            similar_entities_distances=np.array([[1.1], [2.2]], dtype=np.float32),
        )
        similar_entities_producer = SimilarEntitiesProducer(self.default_config)
        self.assertAllEqual(
            [[1], [0], [1]], similar_entities_producer.produce_similar_entities(entity_ids=[0, 1, 0])
        )
        open_mock.assert_called_once_with("/tmp/test_dataset_OUTPUT_RELATIONS_max4.pickle", mode="rb")
        load_mock.assert_called_once_with(open_mock.return_value.__enter__())
        self.assertEqual(load_mock.return_value, similar_entities_producer.similar_entities_storage)
