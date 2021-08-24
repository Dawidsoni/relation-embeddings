from abc import ABCMeta
import numpy as np
import gin.tf

from datasets import dataset_utils
from datasets.raw_dataset import RawDataset
from layers.embeddings_layers import ObjectType


class SoftmaxDataset(RawDataset, metaclass=ABCMeta):
    pass


@gin.configurable
class MaskedEntityOfEdgeDataset(SoftmaxDataset):

    def __init__(self, **kwargs):
        super(MaskedEntityOfEdgeDataset, self).__init__(**kwargs)

    def _generate_samples(self, mask_index):
        for head_id, relation_id, tail_id in self.graph_edges:
            edge_ids = [head_id, relation_id, tail_id]
            output_index = edge_ids[mask_index]
            object_ids = [head_id, relation_id, tail_id]
            object_ids[mask_index] = dataset_utils.MASKED_ENTITY_TOKEN_ID
            object_types = list(dataset_utils.EDGE_OBJECT_TYPES)
            object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
            yield {
                "edge_ids": edge_ids,
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": mask_index,
                "true_entity_index": 0 if mask_index == 2 else 2,
                "expected_output": output_index,
            }

    @property
    def samples(self):
        head_samples = dataset_utils.iterator_of_samples_to_dataset(self._generate_samples(mask_index=0))
        tail_samples = dataset_utils.iterator_of_samples_to_dataset(self._generate_samples(mask_index=2))
        return self._get_processed_dataset(dataset_utils.interleave_datasets(head_samples, tail_samples))


@gin.configurable
class MaskedEntityOfPathDataset(SoftmaxDataset):

    def __init__(self, max_samples=6_000_000, max_draws=18_000_000, **kwargs):
        super(MaskedEntityOfPathDataset, self).__init__(**kwargs)
        self.max_samples = max_samples
        self.max_draws = max_draws

    def _maybe_sample_edge(self, entity_id):
        edges_count = len(self.entity_output_edges[entity_id])
        if edges_count == 0:
            return None
        return self.entity_output_edges[entity_id][np.random.choice(edges_count)]

    def _generate_samples(self, mask_index):
        entities_sampler = dataset_utils.get_int_random_variables_iterator(low=0, high=self.entities_count)
        generated_samples = set()
        for _ in range(self.max_draws):
            if len(generated_samples) >= self.max_samples:
                break
            head_id = next(entities_sampler)
            intermediate_edge = self._maybe_sample_edge(head_id)
            if intermediate_edge is None:
                continue
            intermediate_entity_id, relation1_id = intermediate_edge
            tail_edge = self._maybe_sample_edge(intermediate_entity_id)
            if tail_edge is None:
                continue
            tail_id, relation2_id = tail_edge
            if tail_id == head_id:
                continue
            object_ids = [head_id, tail_id, relation1_id, relation2_id]
            output_index = object_ids[mask_index]
            object_types = [
                ObjectType.ENTITY.value, ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.RELATION.value
            ]
            object_ids[mask_index] = dataset_utils.MASKED_ENTITY_TOKEN_ID
            object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
            generated_samples.add(tuple(object_ids))
            yield {
                "edge_ids": [head_id, dataset_utils.MISSING_RELATION_TOKEN_ID, tail_id],
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": mask_index,
                "true_entity_index": 0 if mask_index == 1 else 1,
                "expected_output": output_index,
                "positions": [0, 2, 3, 4],
            }

    @property
    def samples(self):
        head_samples = dataset_utils.iterator_of_samples_to_dataset(self._generate_samples(mask_index=0))
        tail_samples = dataset_utils.iterator_of_samples_to_dataset(self._generate_samples(mask_index=1))
        return self._get_processed_dataset(dataset_utils.interleave_datasets(head_samples, tail_samples))


"""@gin.configurable
class MaskedAllNeighboursDataset(MaskedEntityOfEdgeDataset):

    def __init__(self, filter_repeated_samples=False, **kwargs):
        super(MaskedAllNeighboursDataset, self).__init__(**kwargs)
        self.filter_repeated_samples = filter_repeated_samples

    def _generate_samples(self, mask_index):
        if self.dataset_type != DatasetType.TRAINING:
            samples_generator = super()._generate_samples(mask_index)
            for sample in samples_generator:
                yield sample
            return
        entity_edges = self.entity_output_edges if mask_index == 2 else self.entity_input_edges
        for source_entity_id, source_edges in entity_edges.items():
            relation_entities = collections.defaultdict(list)
            for destination_entity_id, relation_id in source_edges:
                relation_entities[relation_id].append(destination_entity_id)
            for relation_id, destination_entities in relation_entities.items():
                expected_output = np.zeros(shape=(self.entities_count,))
                for destination_entity_id in destination_entities:
                    expected_output[destination_entity_id] = 1.0
                head_id = source_entity_id if mask_index == 2 else destination_entities[0]
                tail_id = source_entity_id if mask_index == 0 else destination_entities[0]
                edge_ids = [head_id, relation_id, tail_id]
                object_ids = [head_id, relation_id, tail_id]
                object_ids[mask_index] = MASKED_ENTITY_TOKEN_ID
                object_types = list(self.EDGE_OBJECT_TYPES)
                object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
                yield {
                    "edge_ids": edge_ids,
                    "object_ids": object_ids,
                    "object_types": object_types,
                    "mask_index": mask_index,
                    "true_entity_index": 0 if mask_index == 2 else 2,
                    "expected_output": expected_output,
                }


@gin.configurable
class MaskedEntityWithNeighboursDataset(MaskedEntityOfEdgeDataset):
    NEIGHBOUR_OBJECT_TYPES = (ObjectType.ENTITY.value, ObjectType.RELATION.value)

    def __init__(
        self, dataset_type, data_directory, batch_size, neighbours_per_sample, shuffle_dataset=False, **kwargs
    ):
        super(MaskedEntityWithNeighboursDataset, self).__init__(
            dataset_type, data_directory, batch_size, shuffle_dataset, **kwargs
        )
        self.neighbours_per_sample = neighbours_per_sample

    def _produce_object_ids_with_types(self, edges, mask_indexes):
        object_ids, object_types = [], []
        for edge, mask_index in zip(edges.numpy(), mask_indexes.numpy()):
            head_id, relation_id, tail_id = edge
            if mask_index == 0:
                sampled_edges, missing_edges_count = _sample_edges(
                    self.known_entity_input_edges[tail_id],
                    banned_edges=[(head_id, relation_id)],
                    neighbours_per_sample=self.neighbours_per_sample,
                )
            elif mask_index == 2:
                sampled_edges, missing_edges_count = _sample_edges(
                    self.known_entity_output_edges[head_id],
                    banned_edges=[(tail_id, relation_id)],
                    neighbours_per_sample=self.neighbours_per_sample,
                )
            else:
                raise ValueError(f"Expected mask index to be contained in the set {{0, 2}}, got: {mask_index}")
            edge_ids = [head_id, relation_id, tail_id]
            edge_ids[mask_index] = MASKED_ENTITY_TOKEN_ID
            object_ids.append(edge_ids + sampled_edges)
            neighbours_types = list(np.concatenate((
                np.tile(self.NEIGHBOUR_OBJECT_TYPES, reps=self.neighbours_per_sample - missing_edges_count),
                np.tile(ObjectType.SPECIAL_TOKEN.value, reps=2 * missing_edges_count),
            )))
            masked_edge_object_types = list(self.EDGE_OBJECT_TYPES)
            masked_edge_object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
            object_types.append(masked_edge_object_types + neighbours_types)
        return np.array(object_ids), np.array(object_types)

    def _produce_positions(self, mask_indexes):
        list_of_positions = []
        for mask_index in mask_indexes.numpy():
            if mask_index not in [0, 2]:
                raise ValueError(f"Expected mask index to be contained in the set {{0, 2}}, got: {mask_index}")
            neighbour_positions = (3, 4) if mask_index == 0 else (5, 6)
            positions = [0, 1, 2]
            positions.extend(itertools.chain(*[neighbour_positions for _ in range(self.neighbours_per_sample)]))
            list_of_positions.append(positions)
        return np.array(list_of_positions, dtype=np.int32)

    def _map_batched_samples(self, samples):
        object_ids, object_types = tf.py_function(
            self._produce_object_ids_with_types,
            inp=[samples["edge_ids"], samples["mask_index"]],
            Tout=(tf.int32, tf.int32),
        )
        positions = tf.py_function(
            self._produce_positions, inp=[samples["mask_index"]], Tout=tf.int32
        )
        updated_samples = {
            "object_ids": object_ids,
            "object_types": object_types,
            "positions": positions,
        }
        for key, values in samples.items():
            if key in updated_samples:
                continue
            updated_samples[key] = values
        return updated_samples

    @property
    def samples(self):
        edge_samples = super(MaskedEntityWithNeighboursDataset, self).samples
        return edge_samples.map(self._map_batched_samples)
"""