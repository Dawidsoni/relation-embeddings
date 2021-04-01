import numpy as np


class EdgesProducer(object):

    def __init__(self, ids_of_entities, graph_edges):
        self.ids_of_entities = ids_of_entities
        self.set_of_graph_edges = set(graph_edges)

    def _produce_edges(self, edge_pattern, target_pattern_index, swap_index):
        edges_candidates = np.tile(edge_pattern, (len(self.ids_of_entities), 1))
        edges_candidates[:, swap_index] = self.ids_of_entities
        existing_edges_indexes = [
            index for index, candidate in enumerate(edges_candidates)
            if tuple(candidate) in self.set_of_graph_edges and tuple(candidate) != tuple(edge_pattern)
        ]
        filtered_edges = np.delete(edges_candidates, existing_edges_indexes, axis=0)
        pattern_index = np.where((edge_pattern == filtered_edges).all(axis=1))[0][0]
        filtered_edges[[pattern_index, target_pattern_index]] = filtered_edges[[target_pattern_index, pattern_index]]
        return filtered_edges

    def produce_head_edges(self, edge_pattern, target_pattern_index):
        return self._produce_edges(edge_pattern, target_pattern_index, swap_index=0)

    def produce_tail_edges(self, edge_pattern, target_pattern_index):
        return self._produce_edges(edge_pattern, target_pattern_index, swap_index=2)
