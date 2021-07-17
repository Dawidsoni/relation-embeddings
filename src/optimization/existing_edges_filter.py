import numpy as np


class ExistingEdgesFilter(object):

    def __init__(self, entities_count, graph_edges):
        self.entities_count = entities_count
        self.set_of_graph_edges = set(graph_edges)

    def get_values_corresponding_to_existing_edges(self, edge_ids, mask_index, values):
        output_index = edge_ids[mask_index]
        candidate_edges = np.tile(edge_ids, (self.entities_count, 1))
        candidate_edges[:, mask_index] = np.arange(self.entities_count, dtype=np.int32)
        edges_to_keep_indexes = [
            index for index, edge_ids in enumerate(candidate_edges)
            if tuple(edge_ids) not in self.set_of_graph_edges and index != output_index
        ]
        filtered_values = np.concatenate((values[edges_to_keep_indexes], [values[output_index]]))
        target_index = np.random.randint(len(filtered_values))
        filtered_values[-1], filtered_values[target_index] = filtered_values[target_index], filtered_values[-1]
        return filtered_values, target_index
