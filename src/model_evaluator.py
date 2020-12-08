import tensorflow as tf

from edges_producer import EdgesProducer
from evaluation_metrics import EvaluationMetrics


class ModelEvaluator(object):

    def __init__(self, model, loss_object, dataset, existing_graph_edges, output_directory, samples_per_step=100):
        self.model = model
        self.loss_object = loss_object
        self.edges_producer = EdgesProducer(dataset.ids_of_entities, existing_graph_edges)
        self.summary_writer = tf.summary.create_file_writer(output_directory)
        self.iterator_of_samples = iter(dataset.pairs_of_samples.batch(samples_per_step))

    @staticmethod
    def _report_computed_evaluation_metrics(evaluation_metrics):
        mean_rank, mean_reciprocal_rank, hits10 = evaluation_metrics.result()
        tf.summary.scalar(name="mean_rank", data=mean_rank)
        tf.summary.scalar(name="mean_reciprocal_rank", data=mean_reciprocal_rank)
        tf.summary.scalar(name="hits10", data=hits10)

    def _compute_and_report_metrics(self, samples):
        head_metrics = EvaluationMetrics()
        tail_metrics = EvaluationMetrics()
        for edge_sample in samples:
            head_edges = self.edges_producer.produce_head_edges(edge_sample.numpy(), target_pattern_index=0)
            head_losses = self.model(head_edges)
            head_metrics.update_state(head_losses, positive_sample_index=0)
            tail_edges = self.edges_producer.produce_tail_edges(edge_sample.numpy(), target_pattern_index=0)
            tail_losses = self.model(tail_edges)
            tail_metrics.update_state(tail_losses, positive_sample_index=0)
        self._report_computed_evaluation_metrics(head_metrics)
        self._report_computed_evaluation_metrics(tail_metrics)
        average_evaluation_metrics = EvaluationMetrics.get_average_metrics(head_metrics, tail_metrics)
        self._report_computed_evaluation_metrics(average_evaluation_metrics)

    def _compute_and_report_losses(self, positive_samples, negative_samples):
        positive_outputs = self.model(positive_samples)
        negative_outputs = self.model(negative_samples)
        positive_samples_loss = tf.reduce_mean(self.loss_object.get_losses_of_positive_samples(positive_outputs))
        tf.summary.scalar(name="positive_samples_loss", data=positive_samples_loss)
        pairs_of_samples_loss = self.loss_object.get_mean_loss_of_pairs(positive_outputs, negative_outputs)
        tf.summary.scalar(name="pairs_of_samples_loss", data=pairs_of_samples_loss)
        regularization_loss = self.loss_object.get_regularization_loss(self.model)
        tf.summary.scalar(name="regularization_loss", data=regularization_loss)

    def evaluation_step(self):
        positive_samples, negative_samples = next(self.iterator_of_samples)
        with self.summary_writer.as_default():
            self._compute_and_report_metrics(positive_samples)
            self._compute_and_report_losses(positive_samples, negative_samples)

