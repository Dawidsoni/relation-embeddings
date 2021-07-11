from abc import abstractmethod
from typing import Optional, List, Tuple, Union
import tensorflow as tf

from models.knowledge_completion_model import KnowledgeCompletionModel
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


def _extract_lower_is_better_metrics(metrics):
    mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
    return mean_rank, 1.0 - mean_reciprocal_rank, 1.0 - hits10


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

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: SamplingLossObject,
        dataset: SamplingEdgeDataset,
        existing_graph_edges: List[Tuple],
        output_directory: str,
        learning_rate_scheduler: Optional[LearningRateSchedule] = None,
    ):
        super(SamplingModelEvaluator, self).__init__(
            model=model,
            loss_object=loss_object,
            dataset=dataset,
            output_directory=output_directory,
            learning_rate_scheduler=learning_rate_scheduler,
        )
        self.edges_producer = EdgesProducer(dataset.ids_of_entities, existing_graph_edges)

    def _compute_metrics_on_samples(self, batched_samples):
        head_metrics = EvaluationMetrics()
        tail_metrics = EvaluationMetrics()
        for sample in _unbatch_samples(batched_samples):
            head_samples = self.edges_producer.produce_head_edges(sample, target_pattern_index=0)
            head_predictions = self.model.predict(head_samples, batch_size=self.EVAL_BATCH_SIZE)
            head_losses = self.loss_object.get_losses_of_positive_samples(head_predictions)
            head_metrics.update_state(head_losses.numpy(), positive_sample_index=0)
            tail_samples = self.edges_producer.produce_tail_edges(sample, target_pattern_index=0)
            tail_predictions = self.model.predict(tail_samples, batch_size=self.EVAL_BATCH_SIZE)
            tail_losses = self.loss_object.get_losses_of_positive_samples(tail_predictions)
            tail_metrics.update_state(tail_losses.numpy(), positive_sample_index=0)
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
        return _extract_lower_is_better_metrics(metrics["metrics_averaged"])

    def log_metrics(self, logger):
        positive_samples_iterator = self.dataset.samples.map(
            lambda positive_samples, negative_samples: positive_samples
        ).unbatch().batch(len(self.dataset.graph_edges)).as_numpy_iterator()
        named_metrics = self._compute_metrics_on_samples(batched_samples=next(positive_samples_iterator))
        _log_dict_of_metrics(logger, named_metrics)


class SoftmaxModelEvaluator(ModelEvaluator):

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: SupervisedLossObject,
        dataset: SoftmaxDataset,
        existing_graph_edges: List[Tuple],
        output_directory: str,
        learning_rate_scheduler: Optional[LearningRateSchedule] = None,
    ):
        super(SoftmaxModelEvaluator, self).__init__(
            model=model,
            loss_object=loss_object,
            dataset=dataset,
            output_directory=output_directory,
            learning_rate_scheduler=learning_rate_scheduler,
        )
        self.existing_edges_filter = ExistingEdgesFilter(dataset.entities_count, existing_graph_edges)

    def _compute_metrics_on_samples(self, batched_samples):
        head_metrics = EvaluationMetrics()
        tail_metrics = EvaluationMetrics()
        losses_ranking = 1.0 - self.model(batched_samples, training=False).numpy()
        for sample, ranking in zip(_unbatch_samples(batched_samples), losses_ranking):
            filtered_ranking = self.existing_edges_filter.get_values_corresponding_to_existing_edges(
                sample["edge_ids"], sample["mask_index"], values=ranking, target_index=0,
            )
            if sample["mask_index"] == 0:
                head_metrics.update_state(filtered_ranking, positive_sample_index=0)
            elif sample["mask_index"] == 2:
                tail_metrics.update_state(filtered_ranking, positive_sample_index=0)
            else:
                raise ValueError(f"Invalid `mask_index`: {sample['mask_index']}")
        average_evaluation_metrics = EvaluationMetrics.get_average_metrics(head_metrics, tail_metrics)
        return {
            "metrics_head": head_metrics,
            "metrics_tail": tail_metrics,
            "metrics_averaged": average_evaluation_metrics
        }

    def _compute_and_report_losses(self, samples, step):
        predictions = self.model(samples, training=False)
        loss_value = self.loss_object.get_mean_loss_of_samples(
            true_labels=samples["one_hot_output"], predictions=predictions
        )
        tf.summary.scalar(name="losses/samples_loss", data=loss_value, step=step)
        regularization_loss = self.loss_object.get_regularization_loss(self.model)
        tf.summary.scalar(name="losses/regularization_loss", data=regularization_loss, step=step)

    def build_model(self):
        self.model(next(self.iterator_of_samples), training=False)

    def evaluation_step(self, step):
        samples = next(self.iterator_of_samples)
        with self.summary_writer.as_default():
            metrics = self._compute_and_report_metrics(samples, step)
            self._compute_and_report_losses(samples, step)
            self._maybe_report_learning_rate(step)
        return _extract_lower_is_better_metrics(metrics["metrics_averaged"])

    def log_metrics(self, logger):
        test_samples_count = 2 * len(self.dataset.graph_edges)
        samples_iterator = self.dataset.samples.unbatch().batch(test_samples_count).as_numpy_iterator()
        named_metrics = self._compute_metrics_on_samples(batched_samples=next(samples_iterator))
        _log_dict_of_metrics(logger, named_metrics)
