import numpy as np


class LossesFilter(object):

    def __init__(self, ids_of_entities, graph_edges):
        self.ids_of_entities = ids_of_entities
        self.set_of_graph_edges = set(graph_edges)

    def __call__(self, losses, source_position, target_pattern_index):
        banned_entities_ids = [
            index for index, candidate in enumerate(candidate_edges_object_ids)
            if tuple(candidate) in self.set_of_graph_edges and tuple(candidate) != tuple(object_ids)
        ]
        return self._produce_edges(object_ids, object_types, target_pattern_index, swap_index=0)
