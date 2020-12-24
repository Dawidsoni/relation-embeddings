import tensorflow as tf

from edges_producer import EdgesProducer
from evaluation_metrics import EvaluationMetrics


class ModelEvaluator(object):

    def __init__(self, model, loss_object, dataset, existing_graph_edges, output_directory, samples_per_step=250):
        self.model = model
        self.loss_object = loss_object
        self.edges_producer = EdgesProducer(dataset.ids_of_entities, existing_graph_edges)
        self.summary_writer = tf.summary.create_file_writer(output_directory)
        self.iterator_of_samples = dataset.pairs_of_samples.batch(samples_per_step).as_numpy_iterator()

    @staticmethod
    def _report_computed_evaluation_metrics(evaluation_metrics, step, metrics_prefix):
        mean_rank, mean_reciprocal_rank, hits10 = evaluation_metrics.result()
        tf.summary.scalar(name=f"{metrics_prefix}/mean_rank", data=mean_rank, step=step)
        tf.summary.scalar(name=f"{metrics_prefix}/mean_reciprocal_rank", data=mean_reciprocal_rank, step=step)
        tf.summary.scalar(name=f"{metrics_prefix}/hits10", data=hits10, step=step)

    def _compute_and_report_metrics(self, samples, step):
        named_metrics = EvaluationMetrics.compute_metrics_on_samples(
            self.model, self.loss_object, self.edges_producer, samples
        )
        for name_prefix, metrics in named_metrics.items():
            self._report_computed_evaluation_metrics(metrics, step, metrics_prefix=name_prefix)

    def _compute_and_report_losses(self, positive_samples, negative_samples, step):
        positive_outputs = self.model(positive_samples, training=False)
        negative_outputs = self.model(negative_samples, training=False)
        positive_samples_loss = tf.reduce_mean(self.loss_object.get_losses_of_positive_samples(positive_outputs))
        tf.summary.scalar(name="losses/positive_samples_loss", data=positive_samples_loss, step=step)
        pairs_of_samples_loss = self.loss_object.get_mean_loss_of_pairs(positive_outputs, negative_outputs)
        tf.summary.scalar(name="losses/pairs_of_samples_loss", data=pairs_of_samples_loss, step=step)
        regularization_loss = self.loss_object.get_regularization_loss(self.model)
        tf.summary.scalar(name="losses/regularization_loss", data=regularization_loss, step=step)

    def evaluation_step(self, step):
        positive_samples, negative_samples = next(self.iterator_of_samples)
        with self.summary_writer.as_default():
            self._compute_and_report_metrics(positive_samples, step)
            self._compute_and_report_losses(positive_samples, negative_samples, step)

