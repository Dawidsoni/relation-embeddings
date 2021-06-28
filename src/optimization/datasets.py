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


def _get_one_hot_encoded_vector(length, one_hot_ids):
    vector = np.zeros(length)
    vector[one_hot_ids] = 1.0
    return vector


def _get_int_random_variables_iterator(low, high, batch_size=100_000):
    while True:
        for random_variable in np.random.randint(low, high, size=batch_size):
            yield random_variable


def get_outputs_for_sampling_dataset(training_samples, model, training):
    positive_inputs, batched_negative_inputs = training_samples
    positive_outputs = model(positive_inputs, training=training)
    array_of_negative_outputs = []
    for negative_inputs in tf.unstack(tf.stack(batched_negative_inputs), axis=2):
        array_of_negative_outputs.append(model(negative_inputs, training=training))
    return positive_outputs, array_of_negative_outputs


class Dataset(object):
    ENTITIES_IDS_FILENAME = 'entity2id.txt'
    RELATIONS_IDS_FILENAME = 'relation2id.txt'
    TRAINING_DATASET_FILENAME = "train.txt"
    VALIDATION_DATASET_FILENAME = "valid.txt"
    TEST_DATASET_FILENAME = "test.txt"

    def __init__(self, dataset_type, data_directory, batch_size, shuffle_dataset=False):
        self.dataset_type = dataset_type
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        entities_df = pd.read_table(os.path.join(data_directory, self.ENTITIES_IDS_FILENAME), header=None)
        relations_df = pd.read_table(os.path.join(data_directory, self.RELATIONS_IDS_FILENAME), header=None)
        self.entity_ids = dict(zip(entities_df[0], entities_df[1]))
        self.ids_of_entities = list(self.entity_ids.values())
        self.relation_ids = dict(zip(relations_df[0], relations_df[1]))
        self.ids_of_relations = list(self.relation_ids.values())
        self.graph_edges = self._get_graph_edges(incremental_graph=False)
        self.set_of_graph_edges = set(self.graph_edges)
        self.incremental_graph_edges = self._get_graph_edges(incremental_graph=True)
        self.set_of_incremental_graph_edges = set(self.incremental_graph_edges)

    def _extract_edges_from_file(self, dataset_type):
        dataset_types_filenames = {
            DatasetType.TRAINING: Dataset.TRAINING_DATASET_FILENAME,
            DatasetType.VALIDATION: Dataset.VALIDATION_DATASET_FILENAME,
            DatasetType.TEST: Dataset.TEST_DATASET_FILENAME,
        }
        if dataset_type not in dataset_types_filenames:
            raise ValueError(f"Expected an instance of DatasetType, got {self.dataset_type}")
        graph_df = pd.read_table(os.path.join(self.data_directory, dataset_types_filenames[dataset_type]), header=None)
        return list(zip(
            [self.entity_ids[x] for x in graph_df[0]],
            [self.relation_ids[x] for x in graph_df[1]],
            [self.entity_ids[x] for x in graph_df[2]]
        ))

    def _get_graph_edges(self, incremental_graph):
        graph_edges = []
        if self.dataset_type == DatasetType.TRAINING or incremental_graph:
            graph_edges += self._extract_edges_from_file(DatasetType.TRAINING)
        if self.dataset_type == DatasetType.VALIDATION:
            graph_edges += self._extract_edges_from_file(DatasetType.VALIDATION)
        if self.dataset_type == DatasetType.TEST:
            graph_edges += self._extract_edges_from_file(DatasetType.TEST)
        return graph_edges

    def _get_processed_dataset(self, dataset):
        dataset = dataset.shuffle(buffer_size=10_000) if self.shuffle_dataset else dataset
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset.prefetch(100)

    @property
    @abstractmethod
    def samples(self):
        pass


class SamplingDataset(Dataset, metaclass=ABCMeta):
    pass


@gin.configurable(blacklist=['sample_weights_model', 'sample_weights_loss_object'])
class SamplingEdgeDataset(Dataset):
    MAX_ITERATIONS = 1000
    EDGE_OBJECT_TYPES = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]

    def __init__(
        self, dataset_type, data_directory=gin.REQUIRED, batch_size=1, shuffle_dataset=False,
        negatives_per_positive=1, sample_weights_model=None, sample_weights_loss_object=None,
        sample_weights_count=100
    ):
        super(SamplingEdgeDataset, self).__init__(dataset_type, data_directory, batch_size, shuffle_dataset)
        self.negatives_per_positive = negatives_per_positive
        self.sample_weights_model = sample_weights_model
        self.sample_weights_loss_object = sample_weights_loss_object
        self.sample_weights_count = sample_weights_count

    def _get_positive_samples_dataset(self):
        raw_dataset = tf.data.Dataset.from_tensor_slices(self.graph_edges)
        raw_dataset = raw_dataset.map(lambda x: {"object_ids": x, "object_types": self.EDGE_OBJECT_TYPES})
        return self._get_processed_dataset(raw_dataset)

    def _generate_negative_samples(self, negatives_per_positive):
        random_binary_variable_iterator = _get_int_random_variables_iterator(low=0, high=2)
        random_entity_index_iterator = _get_int_random_variables_iterator(low=0, high=max(self.ids_of_entities) + 1)
        for entity_head, relation, entity_tail in self.graph_edges:
            is_head_to_be_swapped = next(random_binary_variable_iterator)
            produced_edges = []
            iterations_count = 0
            while len(produced_edges) < negatives_per_positive and iterations_count < self.MAX_ITERATIONS:
                if is_head_to_be_swapped:
                    entity_head = self.ids_of_entities[next(random_entity_index_iterator)]
                else:
                    entity_tail = self.ids_of_entities[next(random_entity_index_iterator)]
                produced_edge = (entity_head, relation, entity_tail)
                if produced_edge not in self.set_of_graph_edges and produced_edge not in produced_edges:
                    produced_edges.append(produced_edge)
                iterations_count += 1
            if iterations_count < self.MAX_ITERATIONS:
                for produced_edge in produced_edges:
                    yield {
                        "object_ids": produced_edge,
                        "object_types": self.EDGE_OBJECT_TYPES,
                        "head_swapped": is_head_to_be_swapped,
                    }

    def _reorder_negative_samples(self, batched_samples):
        reordered_samples = []
        for key, values in batched_samples.items():
            for index, negative_inputs in enumerate(tf.unstack(values, axis=1)):
                if len(reordered_samples) <= index:
                    reordered_samples.append({})
                reordered_samples[index][key] = negative_inputs
        return reordered_samples

    def _get_negative_samples_dataset(self):
        if self.negatives_per_positive > 1 and self.sample_weights_model is not None:
            raise ValueError("`negatives_per_positive > 1` while `sample_weights_model` is not supported")
        negatives_per_positive = (
            self.negatives_per_positive if self.sample_weights_model is None else self.sample_weights_count
        )
        raw_dataset = tf.data.Dataset.from_generator(
            lambda: self._generate_negative_samples(negatives_per_positive),
            output_signature={"object_ids": tf.TensorSpec(shape=(3, ), dtype=tf.int32),
                              "object_types": tf.TensorSpec(shape=(3,), dtype=tf.int32),
                              "head_swapped": tf.TensorSpec(shape=(), dtype=tf.bool)},
        )
        raw_dataset = raw_dataset.batch(negatives_per_positive, drop_remainder=True)
        return self._get_processed_dataset(raw_dataset).map(self._reorder_negative_samples)

    def _pick_samples_using_model(self, positive_inputs, array_of_negative_inputs):
        positive_outputs = self.sample_weights_model(positive_inputs, training=False)
        array_of_raw_losses = []
        for negative_inputs in array_of_negative_inputs:
            negative_outputs = self.sample_weights_model(negative_inputs, training=False)
            array_of_raw_losses.append(self.sample_weights_loss_object.get_losses_of_pairs(
                positive_outputs, negative_outputs
            ))
        losses = tf.transpose(tf.stack(array_of_raw_losses, axis=0))
        probs = losses / tf.expand_dims(tf.reduce_sum(losses, axis=1), axis=1)
        indexes_of_chosen_samples = tf.reshape(tf.random.categorical(tf.math.log(probs), num_samples=1), (-1, ))
        negative_samples_keys = list(array_of_negative_inputs[0].keys())
        chosen_negative_inputs = {}
        for key in negative_samples_keys:
            stacked_inputs = tf.stack([inputs[key] for inputs in array_of_negative_inputs], axis=1)
            chosen_negative_inputs[key] = tf.gather(stacked_inputs, indexes_of_chosen_samples, axis=1, batch_dims=1)
        return positive_inputs, (chosen_negative_inputs, )

    @property
    def samples(self):
        positive_samples = self._get_positive_samples_dataset()
        negative_samples = self._get_negative_samples_dataset()
        samples = tf.data.Dataset.zip((positive_samples, negative_samples))
        if (self.sample_weights_model is None) != (self.sample_weights_loss_object is None):
            raise ValueError("Expected sample_weights_model and sample_weights_loss_object to be set.")
        if self.sample_weights_model is not None:
            samples = samples.map(self._pick_samples_using_model)
        return samples


class SoftmaxDataset(Dataset, metaclass=ABCMeta):
    pass


@gin.configurable
class MaskedEntityDataset(SoftmaxDataset):

    def __init__(
        self, dataset_type, data_directory=gin.REQUIRED, batch_size=1, shuffle_dataset=False
    ):
        super(MaskedEntityDataset, self).__init__(
            dataset_type, data_directory, batch_size, shuffle_dataset
        )

    def _edge_to_output(self, edge, mask_index):
        object_id = edge[mask_index]
        if self.evaluation_mode:
            return object_id
        return _get_one_hot_encoded_vector(length=max(self.ids_of_entities) + 1, one_hot_ids=[object_id])

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
            outputs.append(_get_one_hot_encoded_vector(
                length=max(self.ids_of_entities) + 1, one_hot_ids=raw_edge[mask_index]
            ))
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
        return merged_samples
