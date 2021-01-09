import numpy as np


class EvaluationMetrics(object):
    HITS10_MAX_POSITION = 10

    def __init__(self):
        super(EvaluationMetrics, self).__init__()
        self.sum_rank = 0.0
        self.sum_reciprocal_rank = 0.0
        self.sum_hits10 = 0.0
        self.samples_count = 0

    @staticmethod
    def get_average_metrics(metrics1, metrics2):
        average_metrics = EvaluationMetrics()
        average_metrics.sum_rank = metrics1.sum_rank + metrics2.sum_rank
        average_metrics.sum_reciprocal_rank = metrics1.sum_reciprocal_rank + metrics2.sum_reciprocal_rank
        average_metrics.sum_hits10 = metrics1.sum_hits10 + metrics2.sum_hits10
        average_metrics.samples_count = metrics1.samples_count + metrics2.samples_count
        return average_metrics

    @staticmethod
    def compute_metrics_on_samples(model, loss_object, edges_producer, samples, batch_size=10_000):
        head_metrics = EvaluationMetrics()
        tail_metrics = EvaluationMetrics()
        for edge_sample in samples:
            head_edges = edges_producer.produce_head_edges(edge_sample, target_pattern_index=0)
            head_losses = loss_object.get_losses_of_positive_samples(model.predict(head_edges, batch_size=batch_size))
            head_metrics.update_state(head_losses.numpy(), positive_sample_index=0)
            tail_edges = edges_producer.produce_tail_edges(edge_sample, target_pattern_index=0)
            tail_losses = loss_object.get_losses_of_positive_samples(model.predict(tail_edges, batch_size=batch_size))
            tail_metrics.update_state(tail_losses.numpy(), positive_sample_index=0)
        average_evaluation_metrics = EvaluationMetrics.get_average_metrics(head_metrics, tail_metrics)
        return {
            "metrics_head": head_metrics,
            "metrics_tail": tail_metrics,
            "metrics_averaged": average_evaluation_metrics
        }

    def update_state(self, losses, positive_sample_index):
        sample_rank = np.where(np.argsort(losses) == positive_sample_index)[0][0] + 1.0
        self.sum_rank += sample_rank
        self.sum_reciprocal_rank += 1.0 / sample_rank
        self.sum_hits10 += (sample_rank <= self.HITS10_MAX_POSITION)
        self.samples_count += 1.0

    def result(self):
        samples_count = max(self.samples_count, 1)
        mean_rank = self.sum_rank / samples_count
        mean_reciprocal_rank = self.sum_reciprocal_rank / samples_count
        hits10 = self.sum_hits10 / samples_count
        return mean_rank, mean_reciprocal_rank, hits10

    def reset_states(self):
        self.sum_rank = 0.0
        self.sum_reciprocal_rank = 0.0
        self.sum_hits10 = 0.0
        self.samples_count = 0
