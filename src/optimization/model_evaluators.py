from abc import abstractmethod
from typing import Optional, List, Tuple, Union
import tensorflow as tf

from models.knowledge_completion_model import KnowledgeCompletionModel
from optimization.datasets import SamplingDataset, Dataset, SoftmaxDataset
from optimization.edges_producer import EdgesProducer
from optimization.evaluation_metrics import EvaluationMetrics
from optimization.loss_objects import SamplingLossObject, SupervisedLossObject
from optimization.losses_filter import LossesFilter

LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


class ModelEvaluator(object):

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: Union[SamplingLossObject, SupervisedLossObject],
        dataset: Dataset,
        output_directory: str,
        learning_rate_scheduler: Optional[LearningRateSchedule],
        samples_per_step: int,
    ):
        self.model = model
        self.loss_object = loss_object
        self.dataset = dataset
        self.iterator_of_samples = dataset.samples.repeat().batch(samples_per_step).as_numpy_iterator()
        self.output_directory = output_directory
        self._summary_writer = None
        self.learning_rate_scheduler = learning_rate_scheduler

    @property
    def summary_writer(self):
        if self._summary_writer is None:
            self._summary_writer = tf.summary.create_file_writer(self.output_directory)
        return self._summary_writer

    @staticmethod
    def _report_computed_evaluation_metrics(evaluation_metrics, step, metrics_prefix):
        mean_rank, mean_reciprocal_rank, hits10 = evaluation_metrics.result()
        tf.summary.scalar(name=f"{metrics_prefix}/mean_rank", data=mean_rank, step=step)
        tf.summary.scalar(name=f"{metrics_prefix}/mean_reciprocal_rank", data=mean_reciprocal_rank, step=step)
        tf.summary.scalar(name=f"{metrics_prefix}/hits10", data=hits10, step=step)

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
        dataset: SamplingDataset,
        existing_graph_edges: List[Tuple],
        output_directory: str,
        learning_rate_scheduler: Optional[LearningRateSchedule] = None,
        samples_per_step: int = 100,
    ):
        super(SamplingModelEvaluator, self).__init__(
            model=model,
            loss_object=loss_object,
            dataset=dataset,
            output_directory=output_directory,
            learning_rate_scheduler=learning_rate_scheduler,
            samples_per_step=samples_per_step,
        )
        self.edges_producer = EdgesProducer(dataset.ids_of_entities, existing_graph_edges)

    def _compute_metrics_on_samples(self, samples):
        head_metrics = EvaluationMetrics()
        tail_metrics = EvaluationMetrics()
        for sample in samples:
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

    def _compute_and_report_metrics(self, samples, step):
        named_metrics = self._compute_metrics_on_samples(samples)
        for name_prefix, metrics in named_metrics.items():
            self._report_computed_evaluation_metrics(metrics, step, metrics_prefix=name_prefix)
        return named_metrics

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
            metrics = self._compute_and_report_metrics(zip(*positive_samples), step)
            self._compute_and_report_losses(positive_samples, step)
            self._compute_and_report_model_outputs(positive_samples, step)
            self._maybe_report_learning_rate(step)
        return metrics["metrics_averaged"].result()[0]

    def log_metrics(self, logger):
        positive_samples_iterator = self.dataset.samples.map(
            lambda positive_samples, negative_samples: positive_samples
        ).as_numpy_iterator()
        named_metrics = self._compute_metrics_on_samples(positive_samples_iterator)
        for name_prefix, metrics in named_metrics.items():
            mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
            logger.info(f"Evaluating a model on test dataset: {name_prefix}/mean_rank: {mean_rank}")
            logger.info(
                f"Evaluating a model on test dataset: {name_prefix}/mean_reciprocal_rank: {mean_reciprocal_rank}"
            )
            logger.info(f"Evaluating a model on test dataset: {name_prefix}/hits10: {hits10}")


class SoftmaxModelEvaluator(ModelEvaluator):

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: SupervisedLossObject,
        dataset: SoftmaxDataset,
        existing_graph_edges: List[Tuple],
        output_directory: str,
        learning_rate_scheduler: Optional[LearningRateSchedule] = None,
        samples_per_step: int = 100,
    ):
        super(SoftmaxModelEvaluator, self).__init__(
            model=model,
            loss_object=loss_object,
            dataset=dataset,
            output_directory=output_directory,
            learning_rate_scheduler=learning_rate_scheduler,
            samples_per_step=samples_per_step,
        )
        self.losses_filter = LossesFilter(dataset.ids_of_entities, existing_graph_edges)

    def _compute_metrics(self, true_outputs, predicted_outputs, original_edges):
        losses = self.loss_object.get_losses_of_samples(true_outputs, predicted_outputs)
        metrics = EvaluationMetrics()
        for true_entity_id, entity_losses, original_edge in zip(true_outputs, losses, original_edges):
            entity_losses = self.losses_filter(entity_losses, source_position=true_entity_id, target_pattern_index=0)
            metrics.update_state(entity_losses.numpy(), positive_sample_index=0)
        return metrics

    def build_model(self):
        pass # TODO

    def evaluation_step(self, step):
        inputs, true_outputs, ids_of_true_outputs, mask_indexes = next(self.iterator_of_samples)
        edges = inputs[0]
        predicted_outputs = self.model(inputs, training=False)
        with self.summary_writer.as_default():
            metrics = self._compute_metrics(edges, ids_of_true_outputs, mask_indexes, true_outputs, predicted_outputs)
            self._report_computed_evaluation_metrics(metrics, step, metrics_prefix="metrics_averaged")
            loss_value = self.loss_object.get_mean_loss_of_samples(true_outputs, predicted_outputs)
            tf.summary.scalar(name="losses/samples_loss", data=loss_value, step=step)
            self._maybe_report_learning_rate(step)
        # TODO: RETURN VALUE

    def log_metrics(self, logger):
        inputs, true_outputs, ids_of_true_outputs, mask_indexes = next(self.iterator_of_samples)
        edges = inputs[0]
        predicted_outputs = self.model(inputs, training=False)
        metrics = self._compute_metrics(true_outputs, predicted_outputs, original_edges)
        mean_rank, mean_reciprocal_rank, hits10 = metrics.result()
        logger.info(f"Evaluating a model on test dataset: metrics_averaged/mean_rank: {mean_rank}")
        logger.info(
            f"Evaluating a model on test dataset: metrics_averaged/mean_reciprocal_rank: {mean_reciprocal_rank}"
        )
        logger.info(f"Evaluating a model on test dataset: metrics_averaged/hits10: {hits10}")
