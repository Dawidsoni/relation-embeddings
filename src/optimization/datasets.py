import collections
import itertools
import functools
from abc import abstractmethod, ABCMeta
import enum
import os
import numpy as np
import tensorflow as tf
import gin.tf
import pandas as pd

from layers.embeddings_layers import ObjectType


TRAINING_DATASET_FILENAME = "train.txt"
VALIDATION_DATASET_FILENAME = "valid.txt"
TEST_DATASET_FILENAME = "test.txt"
ENTITIES_IDS_FILENAME = 'entity2id.txt'
RELATIONS_IDS_FILENAME = 'relation2id.txt'


class DatasetType(enum.Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


def _extract_entity_ids(data_directory):
    entities_df = pd.read_table(os.path.join(data_directory, ENTITIES_IDS_FILENAME), header=None)
    return dict(zip(entities_df[0], entities_df[1]))


def _extract_relation_ids(data_directory):
    relations_df = pd.read_table(os.path.join(data_directory, RELATIONS_IDS_FILENAME), header=None)
    return dict(zip(relations_df[0], relations_df[1]))


def extract_edges_from_file(entity_ids, relation_ids, data_directory, dataset_type):
    dataset_types_filenames = {
        DatasetType.TRAINING: TRAINING_DATASET_FILENAME,
        DatasetType.VALIDATION: VALIDATION_DATASET_FILENAME,
        DatasetType.TEST: TEST_DATASET_FILENAME,
    }
    if dataset_type not in dataset_types_filenames:
        raise ValueError(f"Expected an instance of DatasetType, got {dataset_type}")
    graph_df = pd.read_table(os.path.join(data_directory, dataset_types_filenames[dataset_type]), header=None)
    return list(zip(
        [entity_ids[x] for x in graph_df[0]],
        [relation_ids[x] for x in graph_df[1]],
        [entity_ids[x] for x in graph_df[2]]
    ))


def _get_one_hot_encoded_vector(length, one_hot_ids):
    vector = np.zeros(length)
    vector[one_hot_ids] = 1.0
    return vector


def _get_int_random_variables_iterator(low, high, batch_size=100_000):
    while True:
        for random_variable in np.random.randint(low, high, size=batch_size):
            yield random_variable


def _interleave_datasets(dataset1, dataset2):
    return tf.data.Dataset.zip((dataset1, dataset2)).flat_map(
        lambda x1, x2: tf.data.Dataset.from_tensors(x1).concatenate(tf.data.Dataset.from_tensors(x2))
    )


def get_existing_graph_edges(data_directory):
    entity_ids = _extract_entity_ids(data_directory)
    relation_ids = _extract_relation_ids(data_directory)
    edges_func = functools.partial(
        extract_edges_from_file, entity_ids=entity_ids, relation_ids=relation_ids, data_directory=data_directory
    )
    return (
        edges_func(dataset_type=DatasetType.TRAINING) +
        edges_func(dataset_type=DatasetType.VALIDATION) +
        edges_func(dataset_type=DatasetType.TEST)
    )


class Dataset(object):
    EDGE_OBJECT_TYPES = (ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value)

    def __init__(self, dataset_type, data_directory, batch_size, shuffle_dataset=False):
        self.dataset_type = dataset_type
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        entity_ids = _extract_entity_ids(data_directory)
        relation_ids = _extract_relation_ids(data_directory)
        self.ids_of_entities = list(entity_ids.values())
        self.entities_count = max(self.ids_of_entities) + 1
        self.ids_of_relations = list(relation_ids.values())
        self.relations_count = max(self.ids_of_relations) + 1
        self.graph_edges = extract_edges_from_file(
            entity_ids, relation_ids, self.data_directory, self.dataset_type
        )
        self.set_of_graph_edges = set(self.graph_edges)
        self.entity_output_edges = self._create_entity_output_edges(self.graph_edges)
        self.entity_input_edges = self._create_entity_input_edges(self.graph_edges)
        known_graph_edges = extract_edges_from_file(
            entity_ids, relation_ids, self.data_directory, dataset_type=DatasetType.TRAINING,
        )
        self.known_entity_output_edges = self._create_entity_output_edges(known_graph_edges)
        self.known_entity_input_edges = self._create_entity_input_edges(known_graph_edges)

    def _create_entity_output_edges(self, edges):
        entity_output_edges = collections.defaultdict(list)
        for edge in edges:
            entity_output_edges[edge[0]].append((edge[2], edge[1]))
        return entity_output_edges

    def _create_entity_input_edges(self, edges):
        entity_input_edges = collections.defaultdict(list)
        for edge in edges:
            entity_input_edges[edge[2]].append((edge[0], edge[1]))
        return entity_input_edges

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
        raw_dataset = raw_dataset.map(lambda x: {"object_ids": x, "object_types": list(self.EDGE_OBJECT_TYPES)})
        return self._get_processed_dataset(raw_dataset)

    def _generate_negative_samples(self, negatives_per_positive):
        random_binary_variable_iterator = _get_int_random_variables_iterator(low=0, high=2)
        random_entity_index_iterator = _get_int_random_variables_iterator(low=0, high=self.entities_count)
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
                        "object_types": list(self.EDGE_OBJECT_TYPES),
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


@gin.configurable
class SamplingNeighboursDataset(SamplingEdgeDataset):
    MISSING_EDGE_ENTITY_ID = 0
    MISSING_EDGE_RELATION_ID = 1
    NEIGHBOUR_OBJECT_TYPES = (ObjectType.ENTITY.value, ObjectType.RELATION.value)

    def __init__(
        self, dataset_type, data_directory, batch_size, neighbours_per_sample, shuffle_dataset=False, **kwargs
    ):
        super(SamplingNeighboursDataset, self).__init__(
            dataset_type, data_directory, batch_size, shuffle_dataset, **kwargs
        )
        self.neighbours_per_sample = neighbours_per_sample

    def _sample_edges(self, edges, banned_edges):
        if len(edges) > self.neighbours_per_sample:
            chosen_indexes = np.random.choice(len(edges), size=self.neighbours_per_sample + 1, replace=False)
            chosen_edges = [edges[index] for index in chosen_indexes if edges[index] not in banned_edges]
            return list(itertools.chain(*chosen_edges[:self.neighbours_per_sample])), 0
        chosen_edges = [edge for edge in edges if edge not in banned_edges]
        missing_edges = [
            (self.MISSING_EDGE_ENTITY_ID, self.MISSING_EDGE_RELATION_ID)
            for _ in range(self.neighbours_per_sample - len(chosen_edges))
        ]
        edges = list(itertools.chain(*chosen_edges)) + list(itertools.chain(*missing_edges))
        return edges, len(missing_edges)

    def _produce_object_ids_with_types(self, edges):
        object_ids, object_types = [], []
        for head_id, relation_id, tail_id in edges.numpy():
            sampled_output_edges, missing_output_edges_count = self._sample_edges(
                self.known_entity_output_edges[head_id], banned_edges=[(tail_id, relation_id)]
            )
            sampled_input_edges, missing_input_edges_count = self._sample_edges(
                self.known_entity_input_edges[tail_id], banned_edges=[(head_id, relation_id)]
            )
            object_ids.append([head_id, relation_id, tail_id] + sampled_output_edges + sampled_input_edges)
            outputs_types = list(np.concatenate((
                np.tile(self.NEIGHBOUR_OBJECT_TYPES, reps=self.neighbours_per_sample - missing_output_edges_count),
                np.tile(ObjectType.SPECIAL_TOKEN.value, reps=2 * missing_output_edges_count),
            )))
            inputs_types = list(np.concatenate((
                np.tile(self.NEIGHBOUR_OBJECT_TYPES, reps=self.neighbours_per_sample - missing_input_edges_count),
                np.tile(ObjectType.SPECIAL_TOKEN.value, reps=2 * missing_input_edges_count),
            )))
            object_types.append(list(self.EDGE_OBJECT_TYPES) + outputs_types + inputs_types)
        return np.array(object_ids), np.array(object_types)

    def _produce_positions(self, samples_count):
        outputs_positions = list(itertools.chain(*[(3, 4) for _ in range(self.neighbours_per_sample)]))
        inputs_positions = list(itertools.chain(*[(5, 6) for _ in range(self.neighbours_per_sample)]))
        positions = [0, 1, 2] + outputs_positions + inputs_positions
        return tf.tile(tf.expand_dims(positions, axis=0), multiples=[samples_count, 1])

    def _include_neighbours_in_edges(self, edges):
        object_ids, object_types = tf.py_function(
            self._produce_object_ids_with_types, inp=[edges["object_ids"]], Tout=(tf.int32, tf.int32)
        )
        updated_edges = {
            "object_ids": object_ids,
            "object_types": object_types,
            "positions": self._produce_positions(samples_count=tf.shape(edges["object_ids"])[0]),
        }
        for key, values in edges.items():
            if key in updated_edges:
                continue
            updated_edges[key] = values
        return updated_edges

    def _map_batched_samples(self, positive_edges, array_of_negative_edges):
        positive_edges = self._include_neighbours_in_edges(positive_edges)
        array_of_negative_edges = tuple([
             self._include_neighbours_in_edges(edges) for edges in array_of_negative_edges
        ])
        return positive_edges, array_of_negative_edges

    @property
    def samples(self):
        edge_samples = super(SamplingNeighboursDataset, self).samples
        return edge_samples.map(self._map_batched_samples)


class SoftmaxDataset(Dataset, metaclass=ABCMeta):
    pass


@gin.configurable
class MaskedEntityOfEdgeDataset(SoftmaxDataset):

    def __init__(
        self, dataset_type, data_directory=gin.REQUIRED, batch_size=1, shuffle_dataset=False
    ):
        super(MaskedEntityOfEdgeDataset, self).__init__(
            dataset_type, data_directory, batch_size, shuffle_dataset
        )

    def _get_sample_specification(self):
        return {
            "edge_ids": tf.TensorSpec(shape=(3, ), dtype=tf.int32),
            "object_ids": tf.TensorSpec(shape=(3, ), dtype=tf.int32),
            "object_types": tf.TensorSpec(shape=(3,), dtype=tf.int32),
            "mask_index": tf.TensorSpec(shape=(), dtype=tf.int32),
            "true_entity_index": tf.TensorSpec(shape=(), dtype=tf.int32),
            "one_hot_output": tf.TensorSpec(shape=(self.entities_count, ), dtype=tf.float32),
            "output_index": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

    def _generate_samples(self, mask_index):
        for head_id, relation_id, tail_id in self.graph_edges:
            edge_ids = [head_id, relation_id, tail_id]
            output_index = edge_ids[mask_index]
            object_ids = [head_id, relation_id, tail_id]
            object_ids[mask_index] = 0
            object_types = list(self.EDGE_OBJECT_TYPES)
            object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
            yield {
                "edge_ids": edge_ids,
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": mask_index,
                "true_entity_index": 0 if mask_index == 2 else 2,
                "one_hot_output": _get_one_hot_encoded_vector(length=self.entities_count, one_hot_ids=[output_index]),
                "output_index": output_index,
            }

    @property
    def samples(self):
        sample_specification = self._get_sample_specification()
        head_samples = tf.data.Dataset.from_generator(
            lambda: self._generate_samples(mask_index=0), output_signature=sample_specification,
        )
        tail_samples = tf.data.Dataset.from_generator(
            lambda: self._generate_samples(mask_index=2), output_signature=sample_specification,
        )
        return self._get_processed_dataset(_interleave_datasets(head_samples, tail_samples))
