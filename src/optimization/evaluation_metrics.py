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
