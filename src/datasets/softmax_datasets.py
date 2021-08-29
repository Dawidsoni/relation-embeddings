import abc
import itertools
from abc import ABCMeta, ABC
import random

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

    def __init__(self, max_samples=10_000_000, share_relation_position=False, **kwargs):
        super(MaskedEntityOfPathDataset, self).__init__(**kwargs)
        self.share_relation_position = share_relation_position
        self.max_samples = max_samples

    def _maybe_sample_edge(self, entity_id):
        edges_count = len(self.entity_output_edges[entity_id])
        if edges_count == 0:
            return None
        return self.entity_output_edges[entity_id][np.random.choice(edges_count)]

    def _generate_samples_with_head(self, head_id, mask_index):
        for intermediate_entity_id, relation1_id in self.entity_output_edges[head_id]:
            for tail_id, relation2_id in self.entity_output_edges[intermediate_entity_id]:
                object_ids = [head_id, tail_id, relation1_id, relation2_id]
                output_index = object_ids[mask_index]
                object_types = [
                    ObjectType.ENTITY.value, ObjectType.ENTITY.value, ObjectType.RELATION.value,
                    ObjectType.RELATION.value
                ]
                object_ids[mask_index] = dataset_utils.MASKED_ENTITY_TOKEN_ID
                object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
                positions = [0, 2, 3, 4]
                if self.share_relation_position and mask_index == 0:
                    positions = [0, 2, 1, 3]
                elif self.share_relation_position and mask_index == 1:
                    positions = [0, 2, 4, 1]
                yield {
                    "edge_ids": [head_id, dataset_utils.MISSING_RELATION_TOKEN_ID, tail_id],
                    "object_ids": object_ids,
                    "object_types": object_types,
                    "mask_index": mask_index,
                    "true_entity_index": 0 if mask_index == 1 else 1,
                    "expected_output": output_index,
                    "positions": positions,
                }

    def _generate_samples(self, max_samples_per_entity, mask_index):
        for head_id in range(self.entities_count):
            samples_generator = self._generate_samples_with_head(head_id, mask_index)
            yield from itertools.islice(samples_generator, max_samples_per_entity)

    @property
    def samples(self):
        max_samples_per_entity = (self.max_samples // (2 * self.entities_count))
        head_samples = dataset_utils.iterator_of_samples_to_dataset(
            self._generate_samples(max_samples_per_entity, mask_index=0)
        )
        tail_samples = dataset_utils.iterator_of_samples_to_dataset(
            self._generate_samples(max_samples_per_entity, mask_index=1)
        )
        return self._get_processed_dataset(dataset_utils.interleave_datasets(head_samples, tail_samples))


@gin.configurable
class MaskedRelationOfEdgeDataset(SoftmaxDataset):

    def __init__(self, **kwargs):
        super(MaskedRelationOfEdgeDataset, self).__init__(**kwargs)

    def _generate_samples(self):
        for head_id, relation_id, tail_id in self.graph_edges:
            edge_ids = [head_id, relation_id, tail_id]
            output_index = self.entities_count + edge_ids[1]
            object_ids = [head_id, relation_id, tail_id]
            object_ids[1] = dataset_utils.MASKED_ENTITY_TOKEN_ID
            object_types = list(dataset_utils.EDGE_OBJECT_TYPES)
            object_types[1] = ObjectType.SPECIAL_TOKEN.value
            yield {
                "edge_ids": edge_ids,
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": 1,
                "true_entity_index": -1,
                "expected_output": output_index,
            }

    @property
    def samples(self):
        raw_samples = dataset_utils.iterator_of_samples_to_dataset(self._generate_samples())
        return self._get_processed_dataset(raw_samples)


@gin.configurable
class NeighboursDataset(SoftmaxDataset, ABC):

    def __init__(
        self, max_neighbours_count=gin.REQUIRED, mask_source_entity_pbty=gin.REQUIRED,
        training_epochs_count=5, **kwargs
    ):
        super(NeighboursDataset, self).__init__(**kwargs)
        self.max_neighbours_count = max_neighbours_count
        self.mask_source_entity_pbty = mask_source_entity_pbty
        self.training_epochs_count = training_epochs_count

    def _sample_source_entity_masked(self):
        if self.inference_mode:
            return False
        return np.random.choice(2, p=[1 - self.mask_source_entity_pbty, self.mask_source_entity_pbty])

    def _sample_neighbours(self, neighbours):
        if len(neighbours) <= self.max_neighbours_count:
            sampled_neighbours = neighbours.copy()
            random.shuffle(sampled_neighbours)
            return sampled_neighbours
        sampled_indexes = set(np.random.choice(len(neighbours), self.max_neighbours_count, replace=False))
        return [neighbour for index, neighbour in enumerate(neighbours) if index in sampled_indexes]

    def _generate_object_ids_and_types(
        self, head_id, relation_id, tail_id, mask_index, list_of_neighbours_missing_contexts
    ):
        object_ids = [head_id, relation_id, tail_id]
        object_ids[mask_index] = dataset_utils.MASKED_ENTITY_TOKEN_ID
        object_types = [ObjectType.ENTITY.value, ObjectType.RELATION.value, ObjectType.ENTITY.value]
        object_types[mask_index] = ObjectType.SPECIAL_TOKEN.value
        for neighbours, missing_context in list_of_neighbours_missing_contexts:
            if missing_context:
                missing_ids = [dataset_utils.MISSING_CONTEXT_ENTITY_ID, dataset_utils.MISSING_CONTEXT_RELATION_ID]
                missing_types = [ObjectType.SPECIAL_TOKEN.value, ObjectType.SPECIAL_TOKEN.value]
                object_ids.extend(missing_ids * self.max_neighbours_count)
                object_types.extend(missing_types * self.max_neighbours_count)
                continue
            neighbours = neighbours.copy()
            if mask_index == 0 and (head_id, relation_id) in neighbours:
                neighbours.remove((head_id, relation_id))
            elif mask_index == 2 and (tail_id, relation_id) in neighbours:
                neighbours.remove((tail_id, relation_id))
            sampled_neighbours = self._sample_neighbours(neighbours)
            for sampled_neighbour in sampled_neighbours:
                object_ids.extend(sampled_neighbour)
                object_types.extend([ObjectType.ENTITY.value, ObjectType.RELATION.value])
            for _ in range(self.max_neighbours_count - len(sampled_neighbours)):
                object_ids.extend([dataset_utils.MISSING_ENTITY_TOKEN_ID, dataset_utils.MISSING_RELATION_TOKEN_ID])
                object_types.extend([ObjectType.SPECIAL_TOKEN.value, ObjectType.SPECIAL_TOKEN.value])
        return object_ids, object_types

    @abc.abstractmethod
    def _generate_samples(self, mask_index):
        pass

    @property
    def samples(self):
        repeat_count = 1 if self.inference_mode else self.training_epochs_count
        list_of_datasets = []
        for _ in range(repeat_count):
            list_of_datasets.append(dataset_utils.iterator_of_samples_to_dataset(
                self._generate_samples(mask_index=0))
            )
            list_of_datasets.append(dataset_utils.iterator_of_samples_to_dataset(
                self._generate_samples(mask_index=2))
            )
        return self._get_processed_dataset(dataset_utils.combine_datasets(list_of_datasets))


@gin.configurable
class InputNeighboursDataset(NeighboursDataset):

    def _generate_samples(self, mask_index):
        true_entity_index = 2 if mask_index == 0 else 0
        for head_id, relation_id, tail_id in self.graph_edges:
            edge_ids = [head_id, relation_id, tail_id]
            output_index = edge_ids[mask_index]
            neighbours = (
                self.known_entity_output_edges[tail_id] if mask_index == 0
                else self.known_entity_input_edges[head_id]
            )
            object_ids, object_types = self._generate_object_ids_and_types(
                head_id, relation_id, tail_id, mask_index, list_of_neighbours_missing_contexts=[(neighbours, False)]
            )
            if self._sample_source_entity_masked():
                object_ids[true_entity_index] = dataset_utils.MISSING_SOURCE_ENTITY_ID
                object_types[true_entity_index] = ObjectType.SPECIAL_TOKEN.value
            yield {
                "edge_ids": edge_ids,
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": mask_index,
                "true_entity_index": true_entity_index,
                "expected_output": output_index,
            }


@gin.configurable
class OutputNeighboursDataset(NeighboursDataset):

    def _generate_samples(self, mask_index):
        true_entity_index = 2 if mask_index == 0 else 0
        for head_id, relation_id, tail_id in self.graph_edges:
            edge_ids = [head_id, relation_id, tail_id]
            output_index = edge_ids[mask_index]
            neighbours = (
                self.known_entity_input_edges[tail_id] if mask_index == 0
                else self.known_entity_output_edges[head_id]
            )
            object_ids, object_types = self._generate_object_ids_and_types(
                head_id, relation_id, tail_id, mask_index, list_of_neighbours_missing_contexts=[(neighbours, False)]
            )
            if self._sample_source_entity_masked():
                object_ids[true_entity_index] = dataset_utils.MISSING_SOURCE_ENTITY_ID
                object_types[true_entity_index] = ObjectType.SPECIAL_TOKEN.value
            yield {
                "edge_ids": edge_ids,
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": mask_index,
                "true_entity_index": true_entity_index,
                "expected_output": output_index,
            }


@gin.configurable
class InputOutputNeighboursDataset(NeighboursDataset):

    def __init__(self, mask_input_context_pbty=gin.REQUIRED, mask_output_context_pbty=gin.REQUIRED, **kwargs):
        super(InputOutputNeighboursDataset, self).__init__(**kwargs)
        self.mask_input_context_pbty = mask_input_context_pbty
        self.mask_output_context_pbty = mask_output_context_pbty

    def _sample_input_context_masked(self):
        if self.inference_mode:
            return False
        return np.random.choice(2, p=[1 - self.mask_input_context_pbty, self.mask_input_context_pbty])

    def _sample_output_context_masked(self):
        if self.inference_mode:
            return False
        return np.random.choice(2, p=[1 - self.mask_output_context_pbty, self.mask_output_context_pbty])

    def _sample_contexts_masked(self):
        contexts_masked = [
            self._sample_input_context_masked(),
            self._sample_output_context_masked(),
            self._sample_source_entity_masked()
        ]
        if all(contexts_masked):
            sampled_context = np.random.choice(3)
            contexts_masked[sampled_context] = False
        return contexts_masked

    def _generate_samples(self, mask_index):
        true_entity_index = 2 if mask_index == 0 else 0
        for head_id, relation_id, tail_id in self.graph_edges:
            edge_ids = [head_id, relation_id, tail_id]
            output_index = edge_ids[mask_index]
            input_neighbours = (
                self.known_entity_output_edges[tail_id] if mask_index == 0
                else self.known_entity_input_edges[head_id]
            )
            output_neighbours = (
                self.known_entity_input_edges[tail_id] if mask_index == 0
                else self.known_entity_output_edges[head_id]
            )
            input_context_masked, output_context_masked, source_entity_masked = self._sample_contexts_masked()
            list_of_neighbours_missing_contexts = [
                (input_neighbours, input_context_masked), (output_neighbours, output_context_masked)
            ]
            object_ids, object_types = self._generate_object_ids_and_types(
                head_id, relation_id, tail_id, mask_index, list_of_neighbours_missing_contexts
            )
            if source_entity_masked:
                object_ids[true_entity_index] = dataset_utils.MISSING_SOURCE_ENTITY_ID
                object_types[true_entity_index] = ObjectType.SPECIAL_TOKEN.value
            yield {
                "edge_ids": edge_ids,
                "object_ids": object_ids,
                "object_types": object_types,
                "mask_index": mask_index,
                "true_entity_index": true_entity_index,
                "expected_output": output_index,
            }
