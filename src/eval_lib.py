import tensorflow as tf
import numpy as np


class EvaluationMetrics(tf.keras.metrics.Metric):

    def __init__(self):
        super(EvaluationMetrics, self).__init__()
        self.sum_rank = self.add_weight(name="sum_rank", initializer=0.0, dtype=tf.float32)
        self.sum_reciprocal_rank = self.add_weight(name="sum_reciprocal_rank", initializer=0.0, dtype=tf.float32)
        self.sum_hits10 = self.add_weight(name="sum_hits10", initializer=0.0, dtype=tf.float32)
        self.samples_count = self.add_weight(name="samples_count", initializer=0, dtype=tf.int32)

    @staticmethod
    def get_average_metrics(metrics1, metrics2):
        average_metrics = EvaluationMetrics()
        average_metrics.sum_rank.assign(metrics1.sum_rank + metrics2.sum_rank)
        average_metrics.sum_reciprocal_rank.assign(metrics1.sum_reciprocal_rank + metrics2.sum_reciprocal_rank)
        average_metrics.sum_hits10.assign(metrics1.sum_hits10 + metrics2.sum_hits10)
        average_metrics.samples_count.assign(metrics1.samples_count + metrics2.samples_count)
        return average_metrics

    def update_state(self, losses, positive_sample_index):
        position_of_sample = tf.where(tf.argsort(losses) == positive_sample_index)[0, 0]
        self.sum_rank.assign_add(position_of_sample)
        self.sum_reciprocal_rank(1.0 / position_of_sample)
        self.sum_hits10.assign_add(tf.cast(position_of_sample < 10, dtype=tf.int32))

    def result(self):
        samples_count = tf.maximum(self.samples_count, 1)
        mean_rank = self.sum_rank / samples_count
        mean_reciprocal_rank = self.sum_reciprocal_rank / samples_count
        hits10 = self.sum_hits10 / samples_count
        return {'mean_rank': mean_rank, 'mean_reciprocal_rank': mean_reciprocal_rank, 'hits10': hits10}

    def reset_states(self):
        self.sum_rank.assign(0.0)
        self.sum_reciprocal_rank.assign(0.0)
        self.sum_hits10.assign(0.0)
        self.samples_count.assign(0)


class EvalLib(object):

    def __init__(self, model, loss_object, validation_dataset, existing_graph_edges, samples_per_step=100):
        self.model = model
        self.loss_object = loss_object
        self.validation_dataset = validation_dataset
        self.validation_samples_iterator = validation_dataset.positive_samples.batch(samples_per_step).__iter__()
        self.set_of_graph_edges = set(existing_graph_edges)

    def _get_non_existing_edges_including_edge(self, head_relation_tail, swap_index):
        entity_ids = self.validation_dataset.ids_of_entities
        edges_candidates = np.tile(head_relation_tail, (len(entity_ids), 1))
        edges_candidates[:, swap_index] = entity_ids
        existing_edges_indexes = [
            index for index, candidate in enumerate(edges_candidates)
            if tuple(candidate) in self.set_of_graph_edges
        ]
        return np.delete(edges_candidates, existing_edges_indexes, axis=0)

    def _report_metrics(self, metrics):
        pass

    def evaluation_step(self):
        validation_samples = next(self.validation_samples_iterator)
        head_evaluation_metrics = EvaluationMetrics()
        tail_evaluation_metrics = EvaluationMetrics()
        for head_relation_tail in validation_samples:
            head_edges = self._get_non_existing_edges_including_edge(head_relation_tail, swap_index=0)
            head_losses = self.model(head_edges)
            sample_head_index = np.where((head_edges == tuple(head_relation_tail)).all(axis=1))[0][0]
            head_evaluation_metrics.update_state(head_losses, sample_head_index)
            tail_edges = self._get_non_existing_edges_including_edge(head_relation_tail, swap_index=2)
            tail_losses = self.model(tail_edges)
            sample_tail_index = np.where((tail_edges == tuple(head_relation_tail)).all(axis=1))[0][0]
            tail_evaluation_metrics.update_state(tail_losses, sample_tail_index)
        self._report_metrics(head_evaluation_metrics)
        self._report_metrics(tail_evaluation_metrics)
        average_evaluation_metrics = EvaluationMetrics.get_average_metrics(
            head_evaluation_metrics, tail_evaluation_metrics
        )
        self._report_metrics(average_evaluation_metrics)
