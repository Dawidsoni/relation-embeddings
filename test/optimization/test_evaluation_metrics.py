import tensorflow as tf

from data_handlers.evaluation_metrics import EvaluationMetrics


class TestEvaluationMetrics(tf.test.TestCase):

    @staticmethod
    def _add_metrics_samples(metrics):
        metrics.update_state(losses=[3.0, 2.0, 1.0], positive_sample_index=1)
        metrics.update_state(losses=tf.cast(tf.range(11), dtype=tf.float32), positive_sample_index=10)
        metrics.update_state(losses=[1.0, 0.5, 0.0, 2.0, 3.0], positive_sample_index=0)

    def test_metrics(self):
        metrics = EvaluationMetrics()
        self._add_metrics_samples(metrics)
        mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
        self.assertAllClose(16.0 / 3.0, mean_rank)
        self.assertAllClose((0.5 + 1.0 / 3.0 + 1 / 11.0) / 3.0, mean_reciprocal_rank)
        self.assertAllClose(2.0 / 3.0, hits10)

    def test_get_average_metrics(self):
        metrics1 = EvaluationMetrics()
        self._add_metrics_samples(metrics1)
        metrics2 = EvaluationMetrics()
        metrics2.update_state(losses=[2.0, 1.0, 0.0], positive_sample_index=2)
        joint_metrics = EvaluationMetrics.get_average_metrics(metrics1, metrics2)
        mean_rank, mean_reciprocal_rank, hits10 = joint_metrics.result()
        self.assertAllClose(17.0 / 4.0, mean_rank)
        self.assertAllClose((0.5 + 1.0 / 3.0 + 1 / 11.0 + 1.0) / 4.0, mean_reciprocal_rank)
        self.assertAllClose(3.0 / 4.0, hits10)

    def test_reset_states(self):
        metrics = EvaluationMetrics()
        self._add_metrics_samples(metrics)
        metrics.reset_states()
        self._add_metrics_samples(metrics)
        mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
        self.assertAllClose(16.0 / 3.0, mean_rank)
        self.assertAllClose((0.5 + 1.0 / 3.0 + 1 / 11.0) / 3.0, mean_reciprocal_rank)
        self.assertAllClose(2.0 / 3.0, hits10)

    def test_empty_metrics(self):
        metrics = EvaluationMetrics()
        mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
        self.assertAllClose(0.0, mean_rank)
        self.assertAllClose(0.0, mean_reciprocal_rank)
        self.assertAllClose(0.0, hits10)


if __name__ == '__main__':
    tf.test.main()
