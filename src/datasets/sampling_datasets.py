import itertools
from abc import ABCMeta
import numpy as np
import tensorflow as tf
import gin.tf

from datasets.raw_dataset import RawDataset
from datasets import dataset_utils
from layers.embeddings_layers import ObjectType


class SamplingDataset(RawDataset, metaclass=ABCMeta):
    pass


@gin.configurable(blacklist=['sample_weights_model', 'sample_weights_loss_object'])
class SamplingEdgeDataset(RawDataset):
    MAX_ITERATIONS = 1000

    def __init__(self, negatives_per_positive=1, sample_weights_model=None, sample_weights_loss_object=None,
                 sample_weights_count=100, **kwargs):
        super(SamplingEdgeDataset, self).__init__(**kwargs)
        self.negatives_per_positive = negatives_per_positive
        self.sample_weights_model = sample_weights_model
        self.sample_weights_loss_object = sample_weights_loss_object
        self.sample_weights_count = sample_weights_count

    def _get_positive_samples_dataset(self):
        raw_dataset = tf.data.Dataset.from_tensor_slices(self.graph_edges)
        raw_dataset = raw_dataset.map(
            lambda x: {"object_ids": x, "object_types": list(dataset_utils.EDGE_OBJECT_TYPES)}
        )
        return self._get_processed_dataset(raw_dataset)

    def _generate_negative_samples(self, negatives_per_positive):
        random_binary_variable_iterator = dataset_utils.get_int_random_variables_iterator(low=0, high=2)
        random_entity_index_iterator = dataset_utils.get_int_random_variables_iterator(low=0, high=self.entities_count)
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
                        "object_types": list(dataset_utils.EDGE_OBJECT_TYPES),
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
    NEIGHBOUR_OBJECT_TYPES = (ObjectType.ENTITY.value, ObjectType.RELATION.value)

    def __init__(self, neighbours_per_sample, **kwargs):
        super(SamplingNeighboursDataset, self).__init__(**kwargs)
        self.neighbours_per_sample = neighbours_per_sample

    def _produce_object_ids_with_types(self, edges):
        object_ids, object_types = [], []
        for head_id, relation_id, tail_id in edges.numpy():
            sampled_output_edges, missing_output_edges_count = dataset_utils.sample_edges(
                self.known_entity_output_edges[head_id],
                banned_edges=[(tail_id, relation_id)],
                neighbours_per_sample=self.neighbours_per_sample,
            )
            sampled_input_edges, missing_input_edges_count = dataset_utils.sample_edges(
                self.known_entity_input_edges[tail_id],
                banned_edges=[(head_id, relation_id)],
                neighbours_per_sample=self.neighbours_per_sample,
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
            object_types.append(list(dataset_utils.EDGE_OBJECT_TYPES) + outputs_types + inputs_types)
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
