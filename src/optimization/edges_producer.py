import numpy as np


class EdgesProducer(object):

    def __init__(self, ids_of_entities, graph_edges):
        self.ids_of_entities = ids_of_entities
        self.set_of_graph_edges = set(graph_edges)

    def _produce_edges(self, object_ids, object_types, target_pattern_index, swap_index):
        candidate_edges_object_ids = np.tile(object_ids, (len(self.ids_of_entities), 1))
        candidate_edges_object_ids[:, swap_index] = self.ids_of_entities
        existing_edges_object_ids = [
            index for index, candidate in enumerate(candidate_edges_object_ids)
            if tuple(candidate) in self.set_of_graph_edges and tuple(candidate) != tuple(object_ids)
        ]
        filtered_edges_object_ids = np.delete(candidate_edges_object_ids, existing_edges_object_ids, axis=0)
        pattern_index = np.where((object_ids == filtered_edges_object_ids).all(axis=1))[0][0]
        swap_indexes = [pattern_index, target_pattern_index]
        filtered_edges_object_ids[swap_indexes] = filtered_edges_object_ids[swap_indexes[::-1]]
        filtered_edges_object_types = np.broadcast_to(object_types, filtered_edges_object_ids.shape)
        return filtered_edges_object_ids, filtered_edges_object_types

    def produce_head_edges(self, object_ids, object_types, target_pattern_index):
        return self._produce_edges(object_ids, object_types, target_pattern_index, swap_index=0)

    def produce_tail_edges(self, object_ids, object_types, target_pattern_index):
        return self._produce_edges(object_ids, object_types, target_pattern_index, swap_index=2)
