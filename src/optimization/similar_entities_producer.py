from typing import List
from dataclasses import dataclass
from enum import Enum
import gin.tf
import numpy as np
import os
from sklearn.neighbors import BallTree
import pickle


@gin.constants_from_enum
class EntitiesSimilarityType(Enum):
    INPUT_RELATIONS = "INPUT_RELATIONS"
    OUTPUT_RELATIONS = "OUTPUT_RELATIONS"


@gin.configurable
@dataclass
class SimilarEntitiesProducerConfig(object):
    cache_path: str
    dataset_name: str
    graph_edges: List[List[int]]
    entities_count: int
    relations_count: int
    similarity_type: EntitiesSimilarityType
    similar_entities_count: int


@dataclass
class SimilarEntitiesStorage(object):
    similar_entities_ids: np.ndarray
    similar_entities_distances: np.ndarray


def _get_index_matching_row_index_or_last_index(values, row_index):
    filtered_indexes = [index for index, value in enumerate(values) if value == row_index]
    if len(filtered_indexes) == 0:
        return len(values) - 1
    elif len(filtered_indexes) == 1:
        return filtered_indexes[0]
    else:
        raise ValueError("Found more than one `row_index` in row values")


@gin.configurable
class SimilarEntitiesProducer(object):
    MAX_SIMILAR_ENTITIES_COUNT = 50

    def __init__(self, config: SimilarEntitiesProducerConfig):
        self.config = config
        if self.config.similar_entities_count > self.MAX_SIMILAR_ENTITIES_COUNT:
            raise ValueError(f"Expected 'similar_entities_count' to be smaller than {self.MAX_SIMILAR_ENTITIES_COUNT}"
                             f", got {self.config.similar_entities_count}")
        self.similar_entities_storage = None
        self.similar_entities_to_fetch_count = min(self.MAX_SIMILAR_ENTITIES_COUNT + 1, self.config.entities_count)

    def _create_similarity_matrix(self):
        similarity_matrix = np.zeros(shape=(self.config.entities_count, self.config.relations_count), dtype=np.bool)
        for head_id, relation_id, tail_id in self.config.graph_edges:
            if self.config.similarity_type == EntitiesSimilarityType.INPUT_RELATIONS:
                similarity_matrix[tail_id, relation_id] = True
            elif self.config.similarity_type == EntitiesSimilarityType.OUTPUT_RELATIONS:
                similarity_matrix[head_id, relation_id] = True
        return similarity_matrix

    def _create_similar_entities_storage(self):
        similarity_matrix = self._create_similarity_matrix()
        similarity_tree = BallTree(similarity_matrix, metric='jaccard')
        raw_distances, raw_indexes = similarity_tree.query(similarity_matrix, k=self.similar_entities_to_fetch_count)
        distances, indexes = [], []
        for row_index, (row_of_distances, row_of_indexes) in enumerate(zip(raw_distances, raw_indexes)):
            index_to_remove = _get_index_matching_row_index_or_last_index(row_of_indexes, row_index)
            distances.append(np.delete(row_of_distances, [index_to_remove]))
            indexes.append(np.delete(row_of_indexes, [index_to_remove]))
        return SimilarEntitiesStorage(
            similar_entities_distances=np.array(distances, dtype=np.float32),
            similar_entities_ids=np.array(indexes, dtype=np.int32),
        )

    def _load_or_create_similar_entities_storage(self):
        cached_storage_filename = os.path.join(
            self.config.cache_path,
            f"{self.config.dataset_name}_{self.config.similarity_type.value}_max"
            f"{self.similar_entities_to_fetch_count}.pickle"
        )
        if not os.path.exists(cached_storage_filename):
            similar_entities_storage = self._create_similar_entities_storage()
            with open(cached_storage_filename, mode="wb") as file_stream:
                pickle.dump(similar_entities_storage, file_stream)
            return similar_entities_storage
        with open(cached_storage_filename, mode="rb") as file_stream:
            return pickle.load(file_stream)

    def produce_similar_entities(self, entity_ids):
        if self.similar_entities_storage is None:
            self.similar_entities_storage = self._load_or_create_similar_entities_storage()
        return self.similar_entities_storage.similar_entities_ids[entity_ids, :self.config.similar_entities_count]
