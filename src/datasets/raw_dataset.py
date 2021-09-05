import collections
import gin.tf
from abc import abstractmethod

from datasets import dataset_utils
from datasets.dataset_utils import DatasetType


@gin.configurable
class RawDataset(object):

    def __init__(
        self, dataset_id=gin.REQUIRED, dataset_type=gin.REQUIRED, data_directory=gin.REQUIRED, batch_size=gin.REQUIRED,
        inference_mode=gin.REQUIRED, shuffle_dataset=False, prefetched_samples=10, repeat_dataset=True
    ):
        self.dataset_id = dataset_id
        self.dataset_type = dataset_type
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.inference_mode = inference_mode
        self.shuffle_dataset = shuffle_dataset
        self.prefetched_samples = prefetched_samples
        self.repeat_dataset = repeat_dataset
        entity_ids = dataset_utils.extract_entity_ids(data_directory)
        relation_ids = dataset_utils.extract_relation_ids(data_directory)
        self.ids_of_entities = list(entity_ids.values())
        self.entities_count = max(self.ids_of_entities) + 1
        self.ids_of_relations = list(relation_ids.values())
        self.relations_count = max(self.ids_of_relations) + 1
        self.graph_edges = dataset_utils.extract_edges_from_file(
            entity_ids, relation_ids, self.data_directory, self.dataset_type
        )
        self.set_of_graph_edges = set(self.graph_edges)
        self.entity_output_edges = self._create_entity_output_edges(self.graph_edges)
        self.entity_input_edges = self._create_entity_input_edges(self.graph_edges)
        known_graph_edges = dataset_utils.extract_edges_from_file(
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
        if self.repeat_dataset:
            dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        return dataset.prefetch(self.prefetched_samples)

    @property
    @abstractmethod
    def samples(self):
        pass
