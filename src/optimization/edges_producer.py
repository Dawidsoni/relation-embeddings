import numpy as np
import gin.tf


@gin.configurable(whitelist=["use_entities_order"])
class EdgesProducer(object):

    def __init__(self, ids_of_entities, graph_edges, use_entities_order=False):
        self.ids_of_entities = ids_of_entities
        self.set_of_graph_edges = set(graph_edges)
        self.use_entities_order = use_entities_order

    def _produce_edges(self, sample, swap_index):
        candidates_object_ids = np.tile(sample["object_ids"], (len(self.ids_of_entities), 1))
        candidates_object_ids[:, swap_index] = self.ids_of_entities
        real_object_ids = tuple(sample["object_ids"][:3])
        existing_edges_indexes = [
            index for index, object_ids in enumerate(candidates_object_ids)
            if tuple(object_ids[:3]) in self.set_of_graph_edges and tuple(object_ids[:3]) != real_object_ids
        ]
        filtered_edges_object_ids = np.delete(candidates_object_ids, existing_edges_indexes, axis=0)
        pattern_index = np.where((sample["object_ids"] == filtered_edges_object_ids).all(axis=1))[0][0]
        target_index = pattern_index if self.use_entities_order else np.random.randint(len(filtered_edges_object_ids))
        swap_indexes = [pattern_index, target_index]
        filtered_edges_object_ids[swap_indexes] = filtered_edges_object_ids[swap_indexes[::-1]]
        produced_edges = {"object_ids": filtered_edges_object_ids}
        for key, values in sample.items():
            if key in produced_edges.keys():
                continue
            produced_edges[key] = np.broadcast_to(values, (len(filtered_edges_object_ids), ) + np.shape(values))
        return produced_edges, target_index

    def produce_head_edges(self, sample):
        return self._produce_edges(sample, swap_index=0)

    def produce_tail_edges(self, sample):
        return self._produce_edges(sample, swap_index=2)
