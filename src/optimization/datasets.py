from abc import abstractmethod
import enum
import os
import numpy as np
import tensorflow as tf
import gin.tf
import pandas as pd


class DatasetType(enum.Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


class ObjectType(enum.Enum):
    ENTITY = 0
    RELATION = 1
    MASK = 2


class RawDataset(object):
    ENTITIES_IDS_FILENAME = 'entity2id.txt'
    RELATIONS_IDS_FILENAME = 'relation2id.txt'
    TRAINING_DATASET_FILENAME = "train.txt"
    VALIDATION_DATASET_FILENAME = "valid.txt"
    TEST_DATASET_FILENAME = "test.txt"

    def __init__(self, dataset_type, data_directory, batch_size, repeat_samples):
        self.dataset_type = dataset_type
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.repeat_samples = repeat_samples
        entities_df = pd.read_table(os.path.join(data_directory, self.ENTITIES_IDS_FILENAME), header=None)
        relations_df = pd.read_table(os.path.join(data_directory, self.RELATIONS_IDS_FILENAME), header=None)
        graph_df = pd.read_table(os.path.join(data_directory, self._get_graph_edges_filename()), header=None)
        self.entity_ids = dict(zip(entities_df[0], entities_df[1]))
        self.ids_of_entities = list(self.entity_ids.values())
        self.relation_ids = dict(zip(relations_df[0], relations_df[1]))
        self.ids_of_relations = list(self.relation_ids.values())
        self.graph_edges = list(zip(
            [self.entity_ids[x] for x in graph_df[0]],
            [self.relation_ids[x] for x in graph_df[1]],
            [self.entity_ids[x] for x in graph_df[2]]
        ))
        self.set_of_graph_edges = set(self.graph_edges)

    def _get_graph_edges_filename(self):
        if self.dataset_type == DatasetType.TRAINING:
            return RawDataset.TRAINING_DATASET_FILENAME
        elif self.dataset_type == DatasetType.VALIDATION:
            return RawDataset.VALIDATION_DATASET_FILENAME
        elif self.dataset_type == DatasetType.TEST:
            return RawDataset.TEST_DATASET_FILENAME
        else:
            raise ValueError(f"Expected an instance of DatasetType, got {self.dataset_type}")

    def _get_processed_dataset(self, dataset):
        dataset = dataset.batch(self.batch_size) if self.batch_size is not None else dataset
        dataset = dataset.repeat() if self.repeat_samples else dataset
        return dataset.prefetch(100)

    @property
    @abstractmethod
    def samples(self):
        pass


@gin.configurable
class SamplingDataset(RawDataset):

    def __init__(self, dataset_type, data_directory=gin.REQUIRED, batch_size=None, repeat_samples=False):
        super(SamplingDataset, self).__init__(dataset_type, data_directory, batch_size, repeat_samples)

    @staticmethod
    def _get_integer_random_variables_iterator(low, high, batch_size):
        while True:
            for random_variable in np.random.randint(low, high, size=batch_size):
                yield random_variable

    @staticmethod
    def _with_object_types(dataset):
        object_types = tf.data.Dataset.from_tensor_slices([[
            ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value
        ]]).repeat()
        return tf.data.Dataset.zip((dataset, object_types))

    def _get_positive_samples_dataset(self):
        raw_dataset = tf.data.Dataset.from_tensor_slices(self.graph_edges)
        return self._get_processed_dataset(self._with_object_types(raw_dataset))

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

    def _get_negative_samples_dataset(self):
        raw_dataset = tf.data.Dataset.from_generator(
            self._generate_negative_samples, tf.int32, tf.TensorShape([3])
        )
        return self._get_processed_dataset(self._with_object_types(raw_dataset))

    @property
    def samples(self):
        positive_samples = self._get_positive_samples_dataset()
        if self.dataset_type == DatasetType.TRAINING:
            negative_samples = self._get_negative_samples_dataset()
            return tf.data.Dataset.zip((positive_samples, negative_samples))
        elif self.dataset_type in [DatasetType.VALIDATION, DatasetType.TEST]:
            return positive_samples
        else:
            raise ValueError(f"Expected an instance of DatasetType, got {self.dataset_type}")


@gin.configurable
class MaskedDataset(RawDataset):

    def __init__(self, dataset_type, data_directory=gin.REQUIRED, batch_size=None, repeat_samples=False):
        super(MaskedDataset, self).__init__(dataset_type, data_directory, batch_size, repeat_samples)
        #  TODO: implement this dataset type
