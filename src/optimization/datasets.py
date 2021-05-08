from abc import abstractmethod, ABCMeta
import enum
import os
import numpy as np
import tensorflow as tf
import gin.tf
import pandas as pd

from layers.embeddings_layers import ObjectType


class DatasetType(enum.Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


def get_outputs_for_sampling_dataset(training_samples, model, training):
    positive_inputs, batched_negative_inputs = training_samples
    batched_negative_ids, batched_negative_types = batched_negative_inputs
    array_of_negative_ids = tf.unstack(batched_negative_ids, axis=1)
    array_of_negative_types = tf.unstack(batched_negative_types, axis=1)
    positive_outputs = model(positive_inputs, training=training)
    array_of_negative_outputs = []
    for negative_inputs in zip(array_of_negative_ids, array_of_negative_types):
        array_of_negative_outputs.append(model(negative_inputs, training=training))
    return positive_outputs, array_of_negative_outputs


def _get_one_hot_encoded_vector(object_ids, one_hot_ids):
    vector = np.zeros(max(object_ids) + 1)
    vector[one_hot_ids] = 1.0
    return vector


class Dataset(object):
    ENTITIES_IDS_FILENAME = 'entity2id.txt'
    RELATIONS_IDS_FILENAME = 'relation2id.txt'
    TRAINING_DATASET_FILENAME = "train.txt"
    VALIDATION_DATASET_FILENAME = "valid.txt"
    TEST_DATASET_FILENAME = "test.txt"

    def __init__(self, dataset_type, data_directory, batch_size, repeat_samples=False, shuffle_dataset=False):
        self.dataset_type = dataset_type
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.repeat_samples = repeat_samples
        self.shuffle_dataset = shuffle_dataset
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
            return Dataset.TRAINING_DATASET_FILENAME
        elif self.dataset_type == DatasetType.VALIDATION:
            return Dataset.VALIDATION_DATASET_FILENAME
        elif self.dataset_type == DatasetType.TEST:
            return Dataset.TEST_DATASET_FILENAME
        else:
            raise ValueError(f"Expected an instance of DatasetType, got {self.dataset_type}")

    def _get_processed_dataset(self, dataset):
        dataset = dataset.repeat() if self.repeat_samples else dataset
        dataset = dataset.shuffle(buffer_size=10_000) if self.shuffle_dataset else dataset
        dataset = dataset.batch(self.batch_size) if self.batch_size is not None else dataset
        return dataset.prefetch(100)

    @property
    @abstractmethod
    def samples(self):
        pass


@gin.configurable
class SamplingDataset(Dataset):
    MAX_ITERATIONS = 1000

    def __init__(
        self,
        dataset_type,
        data_directory=gin.REQUIRED,
        batch_size=None,
        repeat_samples=False,
        shuffle_dataset=False,
        negatives_per_positive=1,
        sample_weights_model=None,
        weights_candidates_count=100
    ):
        super(SamplingDataset, self).__init__(
            dataset_type, data_directory, batch_size, repeat_samples, shuffle_dataset
        )
        self.negatives_per_positive = negatives_per_positive
        self.sample_weights_model = sample_weights_model
        self.weights_candidates_count = weights_candidates_count

    @staticmethod
    def _get_integer_random_variables_iterator(low, high, batch_size):
        while True:
            for random_variable in np.random.randint(low, high, size=batch_size):
                yield random_variable

    @staticmethod
    def _with_object_types(dataset):
        samples_shape = tf.data.experimental.get_structure(dataset).shape
        object_types = tf.data.Dataset.from_tensor_slices([[
            ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value
        ]]).repeat()
        if len(samples_shape) == 1:
            return tf.data.Dataset.zip((dataset, object_types))
        return tf.data.Dataset.zip((dataset, object_types.batch(samples_shape[0])))

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
            is_head_to_be_swapped = next(random_binary_variable_iterator)
            produced_edges = []
            iterations_count = 0
            while len(produced_edges) < self.negatives_per_positive and iterations_count < self.MAX_ITERATIONS:
                if is_head_to_be_swapped:
                    entity_head = self.ids_of_entities[next(random_entity_index_iterator)]
                else:
                    entity_tail = self.ids_of_entities[next(random_entity_index_iterator)]
                produced_edge = (entity_head, relation, entity_tail)
                if produced_edge not in self.set_of_graph_edges and produced_edge not in produced_edges:
                    produced_edges.append(produced_edge)
                iterations_count += 1
            yield np.array(produced_edges, dtype=np.int32)

    def _sample_negatives_using_model(self, positive_sample, negative_samples):
        positive_outputs, array_of_negative_outputs = get_outputs_for_sampling_dataset(
            (positive_sample, negative_samples), self.sample_weights_model, training=False
        )
        return positive_outputs

    def _get_negative_samples_dataset(self):
        if self.negatives_per_positive > 1 and self.sample_weights_model is not None:
            raise ValueError("Cannot set negatives_per_positive > 1 and sample_weights_model at the same time")
        negatives_per_positive = (
            self.negatives_per_positive if self.sample_weights_model is None else self.weights_candidates_count
        )
        raw_dataset = tf.data.Dataset.from_generator(
            self._generate_negative_samples, tf.int32, tf.TensorShape([negatives_per_positive, 3])
        )
        return self._get_processed_dataset(self._with_object_types(raw_dataset))

    @property
    def samples(self):
        positive_samples = self._get_positive_samples_dataset()
        negative_samples = self._get_negative_samples_dataset()
        samples = tf.data.Dataset.zip((positive_samples, negative_samples))
        if self.sample_weights_model is not None:
            samples = samples.map(self._sample_negatives_using_model)
        return samples


class SoftmaxDataset(Dataset, metaclass=ABCMeta):
    pass


@gin.configurable
class MaskedEntityDataset(SoftmaxDataset):

    def __init__(
        self, dataset_type, data_directory=gin.REQUIRED, batch_size=None, repeat_samples=False, shuffle_dataset=False
    ):
        super(MaskedEntityDataset, self).__init__(
            dataset_type, data_directory, batch_size, repeat_samples, shuffle_dataset
        )

    def _edge_to_output(self, edge, mask_index):
        object_id = edge[mask_index]
        if self.evaluation_mode:
            return object_id
        return _get_one_hot_encoded_vector(self.ids_of_entities, [object_id])

    def _get_input_object_types(self, mask_index):
        object_types = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]
        object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
        return object_types

    def _get_masked_dataset(self, mask_index):
        input_object_ids, outputs, ids_of_outputs = [], [], []
        for raw_edge in self.graph_edges:
            masked_edge = list(raw_edge).copy()
            masked_edge[mask_index] = 0
            input_object_ids.append(tuple(masked_edge))
            outputs.append(_get_one_hot_encoded_vector(self.ids_of_entities, raw_edge[mask_index]))
            ids_of_outputs.append(raw_edge[mask_index])
        input_objects_dataset = tf.data.Dataset.from_tensor_slices(input_object_ids)
        object_types_dataset = tf.data.Dataset.from_tensor_slices([self._get_input_object_types(mask_index)]).repeat()
        inputs_dataset = tf.data.Dataset.zip((input_objects_dataset, object_types_dataset))
        outputs_dataset = tf.data.Dataset.from_tensor_slices(outputs)
        ids_of_outputs_dataset = tf.data.Dataset.from_tensor_slices(ids_of_outputs)
        mask_indexes_dataset = tf.data.Dataset.from_tensor_slices([mask_index]).repeat()
        return tf.data.Dataset.zip((
            self._get_processed_dataset(inputs_dataset),
            self._get_processed_dataset(outputs_dataset),
            self._get_processed_dataset(ids_of_outputs_dataset),
            self._get_processed_dataset(mask_indexes_dataset),
        ))

    @property
    def samples(self):
        head_samples = self._get_masked_dataset(mask_index=0)
        tail_samples = self._get_masked_dataset(mask_index=2)
        merged_samples = head_samples.concatenate(tail_samples)
        if self.shuffle_dataset:
            merged_samples = merged_samples.shuffle(buffer_size=10_000)
        return merged_samples
