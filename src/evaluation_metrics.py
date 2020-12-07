import tensorflow as tf


class EvaluationMetrics(tf.keras.metrics.Metric):
    HITS10_MAX_POSITION = 10

    def __init__(self):
        super(EvaluationMetrics, self).__init__()
        initializer = tf.zeros_initializer()
        self.sum_rank = self.add_weight(name="sum_rank", initializer=initializer, dtype=tf.float32)
        self.sum_reciprocal_rank = self.add_weight(
            name="sum_reciprocal_rank", initializer=initializer, dtype=tf.float32
        )
        self.sum_hits10 = self.add_weight(name="sum_hits10", initializer=initializer, dtype=tf.float32)
        self.samples_count = self.add_weight(name="samples_count", initializer=initializer, dtype=tf.float32)

    @staticmethod
    def get_average_metrics(metrics1, metrics2):
        average_metrics = EvaluationMetrics()
        average_metrics.sum_rank.assign(metrics1.sum_rank + metrics2.sum_rank)
        average_metrics.sum_reciprocal_rank.assign(metrics1.sum_reciprocal_rank + metrics2.sum_reciprocal_rank)
        average_metrics.sum_hits10.assign(metrics1.sum_hits10 + metrics2.sum_hits10)
        average_metrics.samples_count.assign(metrics1.samples_count + metrics2.samples_count)
        return average_metrics

    def update_state(self, losses, positive_sample_index):
        sample_rank = tf.cast(tf.where(tf.argsort(losses) == positive_sample_index)[0, 0], tf.float32) + 1.0
        self.sum_rank.assign_add(sample_rank)
        self.sum_reciprocal_rank.assign_add(1.0 / sample_rank)
        self.sum_hits10.assign_add(tf.cast(sample_rank <= self.HITS10_MAX_POSITION, dtype=tf.float32))
        self.samples_count.assign_add(1.0)

    def result(self):
        samples_count = tf.maximum(self.samples_count, 1.0)
        mean_rank = self.sum_rank / samples_count
        mean_reciprocal_rank = self.sum_reciprocal_rank / samples_count
        hits10 = self.sum_hits10 / samples_count
        return mean_rank, mean_reciprocal_rank, hits10

    def reset_states(self):
        self.sum_rank.assign(0.0)
        self.sum_reciprocal_rank.assign(0.0)
        self.sum_hits10.assign(0.0)
        self.samples_count.assign(0)
