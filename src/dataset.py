import os
import numpy as np
import tensorflow as tf
import gin.tf
import pandas as pd


@gin.configurable
class Dataset(object):
    ENTITIES_IDS_FILENAME = 'entity2id.txt'
    RELATIONS_IDS_FILENAME = 'relation2id.txt'

    def __init__(
        self, graph_edges_filename=gin.REQUIRED, data_directory=gin.REQUIRED, batch_size=None, repeat_samples=False
    ):
        self._load_data(data_directory, graph_edges_filename)
        self.positive_samples = self._get_positive_samples_dataset(batch_size, repeat_samples)
        self.negative_samples = self._get_negative_samples_dataset(batch_size, repeat_samples)
        self.pairs_of_samples = tf.data.Dataset.zip((self.positive_samples, self.negative_samples))

    def _load_data(self, data_directory, graph_edges_filename):
        entities_df = pd.read_table(
            os.path.join(data_directory, self.ENTITIES_IDS_FILENAME), header=None
        )
        self.entity_ids = dict(zip(entities_df[0], entities_df[1]))
        self.ids_of_entities = list(self.entity_ids.values())
        relations_df = pd.read_table(
            os.path.join(data_directory, self.RELATIONS_IDS_FILENAME), header=None
        )
        self.relation_ids = dict(zip(relations_df[0], relations_df[1]))
        self.ids_of_relations = list(self.relation_ids.values())
        graph_df = pd.read_table(
            os.path.join(data_directory, graph_edges_filename), header=None
        )
        self.graph_edges = list(zip(
            [self.entity_ids[x] for x in graph_df[0]],
            [self.relation_ids[x] for x in graph_df[1]],
            [self.entity_ids[x] for x in graph_df[2]]
        ))
        self.set_of_graph_edges = set(self.graph_edges)

    @staticmethod
    def _get_processed_dataset(dataset, batch_size, repeat_samples):
        dataset = dataset.batch(batch_size) if batch_size is not None else dataset
        dataset = dataset.repeat() if repeat_samples else dataset
        return dataset.prefetch(100)

    @staticmethod
    def _get_integer_random_variables_iterator(low, high, batch_size):
        while True:
            for random_variable in np.random.randint(low, high, size=batch_size):
                yield random_variable

    def _get_positive_samples_dataset(self, batch_size=None, repeat_samples=False):
        raw_dataset = tf.data.Dataset.from_tensor_slices(self.graph_edges)
        return self._get_processed_dataset(raw_dataset, batch_size, repeat_samples)

    def _generate_negative_samples(self):
        random_binary_variable_iterator = self._get_integer_random_variables_iterator(
            low=0, high=2, batch_size=100_000
        )
        random_entity_index_iterator = self._get_integer_random_variables_iterator(
            low=0, high=len(self.ids_of_entities), batch_size=100_000
        )
        for entity_head, relation, entity_tail in self.graph_edges:
            swapped_entity_head, swapped_entity_tail = entity_head, entity_tail
            while (swapped_entity_head, relation, swapped_entity_tail) in self.set_of_graph_edges:
                swapped_entity_head, swapped_entity_tail = entity_head, entity_tail
                if next(random_binary_variable_iterator):
                    swapped_entity_head = self.ids_of_entities[next(random_entity_index_iterator)]
                else:
                    swapped_entity_tail = self.ids_of_entities[next(random_entity_index_iterator)]
            yield np.array([swapped_entity_head, relation, swapped_entity_tail], dtype=np.int32)

    def _get_negative_samples_dataset(self, batch_size=None, repeat_samples=False):
        raw_dataset = tf.data.Dataset.from_generator(
            self._generate_negative_samples, tf.int32, tf.TensorShape([3])
        )
        return self._get_processed_dataset(raw_dataset, batch_size, repeat_samples)
