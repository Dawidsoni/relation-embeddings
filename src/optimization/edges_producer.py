import numpy as np


class EdgesProducer(object):

    def __init__(self, ids_of_entities, graph_edges):
        self.ids_of_entities = ids_of_entities
        self.set_of_graph_edges = set(graph_edges)

    def _produce_edges(self, sample, target_pattern_index, swap_index):
        candidates_object_ids = np.tile(sample["object_ids"], (len(self.ids_of_entities), 1))
        candidates_object_ids[:, swap_index] = self.ids_of_entities
        real_object_ids = tuple(sample["object_ids"][:3])
        existing_edges_indexes = [
            index for index, object_ids in enumerate(candidates_object_ids)
            if tuple(object_ids[:3]) in self.set_of_graph_edges and tuple(object_ids[:3]) != real_object_ids
        ]
        filtered_edges_object_ids = np.delete(candidates_object_ids, existing_edges_indexes, axis=0)
        pattern_index = np.where((sample["object_ids"] == filtered_edges_object_ids).all(axis=1))[0][0]
        swap_indexes = [pattern_index, target_pattern_index]
        filtered_edges_object_ids[swap_indexes] = filtered_edges_object_ids[swap_indexes[::-1]]
        produced_edges = {"object_ids": filtered_edges_object_ids}
        for key, values in sample.items():
            if key == "object_ids":
                continue
            produced_edges[key] = np.broadcast_to(values, (len(filtered_edges_object_ids), ) + np.shape(values))
        return produced_edges

    def produce_head_edges(self, sample, target_pattern_index):
        return self._produce_edges(sample, target_pattern_index, swap_index=0)

    def produce_tail_edges(self, sample, target_pattern_index):
        return self._produce_edges(sample, target_pattern_index, swap_index=2)
