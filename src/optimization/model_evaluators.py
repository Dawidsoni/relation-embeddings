from abc import abstractmethod
from typing import Optional, List, Tuple, Union
import tensorflow as tf

from models.knowledge_completion_model import KnowledgeCompletionModel
from optimization import datasets
from optimization.datasets import SamplingEdgeDataset, Dataset, SoftmaxDataset
from optimization.edges_producer import EdgesProducer
from optimization.evaluation_metrics import EvaluationMetrics
from optimization.existing_edges_filter import ExistingEdgesFilter
from optimization.loss_objects import SamplingLossObject, SupervisedLossObject

LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


def _report_computed_evaluation_metrics(evaluation_metrics, step, metrics_prefix):
    mean_rank, mean_reciprocal_rank, hits10 = evaluation_metrics.result()
    tf.summary.scalar(name=f"{metrics_prefix}/mean_rank", data=mean_rank, step=step)
    tf.summary.scalar(name=f"{metrics_prefix}/mean_reciprocal_rank", data=mean_reciprocal_rank, step=step)
    tf.summary.scalar(name=f"{metrics_prefix}/hits10", data=hits10, step=step)


def _unbatch_samples(batched_samples):
    samples_any_key = list(batched_samples.keys())[0]
    batch_size = tf.shape(batched_samples[samples_any_key])[0]
    return [
        {key: values[index] for key, values in batched_samples.items()}
        for index in range(batch_size)
    ]


def _log_dict_of_metrics(logger, dict_of_metrics):
    for name_prefix, metrics in dict_of_metrics.items():
        mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
        logger.info(f"Evaluating a model on test dataset: {name_prefix}/mean_rank: {mean_rank}")
        logger.info(
            f"Evaluating a model on test dataset: {name_prefix}/mean_reciprocal_rank: {mean_reciprocal_rank}"
        )
        logger.info(f"Evaluating a model on test dataset: {name_prefix}/hits10: {hits10}")


class ModelEvaluator(object):

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: Union[SamplingLossObject, SupervisedLossObject],
        dataset: Dataset,
        output_directory: str,
        learning_rate_scheduler: Optional[LearningRateSchedule],
    ):
        self.model = model
        self.loss_object = loss_object
        self.dataset = dataset
        self.iterator_of_samples = dataset.samples.as_numpy_iterator()
        self.output_directory = output_directory
        self._summary_writer = None
        self.learning_rate_scheduler = learning_rate_scheduler

    @property
    def summary_writer(self):
        if self._summary_writer is None:
            self._summary_writer = tf.summary.create_file_writer(self.output_directory)
        return self._summary_writer

    @abstractmethod
    def _compute_metrics_on_samples(self, batched_samples):
        pass

    def _compute_and_report_metrics(self, batched_samples, step):
        named_metrics = self._compute_metrics_on_samples(batched_samples)
        for name_prefix, metrics in named_metrics.items():
            _report_computed_evaluation_metrics(metrics, step, metrics_prefix=name_prefix)
        return named_metrics

    def _maybe_report_learning_rate(self, step):
        if self.learning_rate_scheduler is None:
            return
        learning_rate = self.learning_rate_scheduler(step)
        tf.summary.scalar(name="optimizer/learning_rate", data=learning_rate, step=step)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def evaluation_step(self, step):
        pass

    @abstractmethod
    def log_metrics(self, logger):
        pass


class SamplingModelEvaluator(ModelEvaluator):
    EVAL_BATCH_SIZE = 10_000

    def __init__(self, **kwargs):
        super(SamplingModelEvaluator, self).__init__(**kwargs)
        existing_graph_edges = datasets.get_existing_graph_edges(self.dataset.data_directory)
        self.edges_producer = EdgesProducer(self.dataset.ids_of_entities, existing_graph_edges)

    def _compute_metrics_on_samples(self, batched_samples):
        head_metrics = EvaluationMetrics()
        tail_metrics = EvaluationMetrics()
        for sample in _unbatch_samples(batched_samples):
            head_samples, head_target_index = self.edges_producer.produce_head_edges(sample)
            head_predictions = self.model.predict(head_samples, batch_size=self.EVAL_BATCH_SIZE)
            head_losses = self.loss_object.get_losses_of_positive_samples(head_predictions)
            head_metrics.update_state(head_losses.numpy(), positive_sample_index=head_target_index)
            tail_samples, tail_target_index = self.edges_producer.produce_tail_edges(sample)
            tail_predictions = self.model.predict(tail_samples, batch_size=self.EVAL_BATCH_SIZE)
            tail_losses = self.loss_object.get_losses_of_positive_samples(tail_predictions)
            tail_metrics.update_state(tail_losses.numpy(), positive_sample_index=tail_target_index)
        average_evaluation_metrics = EvaluationMetrics.get_average_metrics(head_metrics, tail_metrics)
        return {
            "metrics_head": head_metrics,
            "metrics_tail": tail_metrics,
            "metrics_averaged": average_evaluation_metrics
        }

    def _compute_and_report_losses(self, positive_samples, step):
        positive_outputs = self.model(positive_samples, training=False)
        positive_samples_loss = tf.reduce_mean(self.loss_object.get_losses_of_positive_samples(positive_outputs))
        tf.summary.scalar(name="losses/positive_samples_loss", data=positive_samples_loss, step=step)
        regularization_loss = self.loss_object.get_regularization_loss(self.model)
        tf.summary.scalar(name="losses/regularization_loss", data=regularization_loss, step=step)
        return positive_samples_loss.numpy()

    def _compute_and_report_model_outputs(self, positive_samples, step):
        positive_outputs = self.model(positive_samples, training=False)
        flat_positive_outputs = tf.reshape(positive_outputs, shape=(-1,))
        positive_norms = tf.norm(positive_outputs, axis=1)
        tf.summary.histogram(name="distributions/positive_samples_outputs", data=flat_positive_outputs, step=step)
        tf.summary.histogram(name="distributions/positive_samples_l2_norms", data=positive_norms, step=step)

    def build_model(self):
        positive_samples, unused_negative_samples = next(self.iterator_of_samples)
        self.model(positive_samples, training=False)

    def evaluation_step(self, step):
        positive_samples, unused_negative_samples = next(self.iterator_of_samples)
        with self.summary_writer.as_default():
            metrics = self._compute_and_report_metrics(positive_samples, step)
            self._compute_and_report_losses(positive_samples, step)
            self._compute_and_report_model_outputs(positive_samples, step)
            self._maybe_report_learning_rate(step)
        return metrics["metrics_averaged"].result()[1]

    def log_metrics(self, logger):
        positive_samples_iterator = self.dataset.samples.map(
            lambda positive_samples, negative_samples: positive_samples
        ).unbatch().batch(len(self.dataset.graph_edges)).as_numpy_iterator()
        named_metrics = self._compute_metrics_on_samples(batched_samples=next(positive_samples_iterator))
        _log_dict_of_metrics(logger, named_metrics)


class SoftmaxModelEvaluator(ModelEvaluator):

    def __init__(self, **kwargs):
        super(SoftmaxModelEvaluator, self).__init__(**kwargs)
        existing_graph_edges = datasets.get_existing_graph_edges(self.dataset.data_directory)
        self.existing_edges_filter = ExistingEdgesFilter(self.dataset.entities_count, existing_graph_edges)

    def _compute_metrics_on_samples(self, list_of_batched_samples):
        raw_head_metrics, raw_tail_metrics = EvaluationMetrics(), EvaluationMetrics()
        head_metrics, tail_metrics = EvaluationMetrics(), EvaluationMetrics()
        for batched_samples in list_of_batched_samples:
            losses_ranking = 1.0 - self.model(batched_samples, training=False).numpy()
            for sample, ranking in zip(_unbatch_samples(batched_samples), losses_ranking):
                filtered_ranking, target_index = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
                    sample["edge_ids"], sample["mask_index"], values=ranking
                )
                if sample["mask_index"] == 0:
                    raw_head_metrics.update_state(ranking, positive_sample_index=sample["expected_output"])
                    head_metrics.update_state(filtered_ranking, positive_sample_index=target_index)
                elif sample["mask_index"] == 2:
                    raw_tail_metrics.update_state(ranking, positive_sample_index=sample["expected_output"])
                    tail_metrics.update_state(filtered_ranking, positive_sample_index=target_index)
                else:
                    raise ValueError(f"Invalid `mask_index`: {sample['mask_index']}")
        raw_averaged_evaluation_metrics = EvaluationMetrics.get_average_metrics(raw_head_metrics, raw_tail_metrics)
        averaged_evaluation_metrics = EvaluationMetrics.get_average_metrics(head_metrics, tail_metrics)
        return {
            "raw_metrics_head": raw_head_metrics,
            "raw_metrics_tail": raw_tail_metrics,
            "raw_metrics_averaged": raw_averaged_evaluation_metrics,
            "metrics_head": head_metrics,
            "metrics_tail": tail_metrics,
            "metrics_averaged": averaged_evaluation_metrics,
        }

    def _compute_and_report_losses(self, list_of_batched_samples, step):
        loss_values = []
        for batched_samples in list_of_batched_samples:
            predictions = self.model(batched_samples, training=False)
            loss_values.extend(self.loss_object.get_losses_of_samples(
                true_labels=batched_samples["expected_output"], predictions=predictions
            ))
        loss_value = tf.reduce_mean(loss_values)
        tf.summary.scalar(name="losses/samples_loss", data=loss_value, step=step)
        regularization_loss = self.loss_object.get_regularization_loss(self.model)
        tf.summary.scalar(name="losses/regularization_loss", data=regularization_loss, step=step)
        return loss_value.numpy()

    def build_model(self):
        self.model(next(self.iterator_of_samples), training=False)

    def evaluation_step(self, step):
        list_of_batched_samples = [next(self.iterator_of_samples), next(self.iterator_of_samples)]
        with self.summary_writer.as_default():
            metrics = self._compute_and_report_metrics(list_of_batched_samples, step)
            self._compute_and_report_losses(list_of_batched_samples, step)
            self._maybe_report_learning_rate(step)
        return metrics["metrics_averaged"].result()[1]

    def log_metrics(self, logger):
        test_samples_count = 2 * len(self.dataset.graph_edges)
        samples_iterator = self.dataset.samples.unbatch().batch(test_samples_count).as_numpy_iterator()
        named_metrics = self._compute_metrics_on_samples(list_of_batched_samples=[next(samples_iterator)])
        _log_dict_of_metrics(logger, named_metrics)
