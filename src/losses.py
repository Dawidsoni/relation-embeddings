from enum import Enum
import tensorflow as tf


class OptimizedMetric(Enum):
    NORM = 1
    SOFTPLUS = 2


class LossObject(object):

    def __init__(self, optimized_metric, norm_metric_order=None, norm_metric_margin=None):
        self.optimized_metric = optimized_metric
        self.norm_metric_order = norm_metric_order
        self.norm_metric_margin = norm_metric_margin

    @staticmethod
    def _get_softplus_losses_of_samples(samples):
        if samples.shape[1] != 1:
            raise ValueError('Softplus metric is incompatible with embeddings of shape greater than 1')
        return tf.math.softplus(samples)

    @staticmethod
    def _get_softplus_mean_loss_of_pairs(positive_samples, negative_samples):
        positive_losses = LossObject._get_softplus_losses_of_samples(positive_samples)
        negative_losses = LossObject._get_softplus_losses_of_samples(-negative_samples)
        return tf.reduce_mean(tf.concat([positive_losses, negative_losses], axis=0)) / 2.0

    def _get_norm_losses_of_samples(self, samples):
        if self.norm_metric_order is None:
            raise ValueError('Norm metric requires to define norm_metric_order')
        return tf.norm(samples, axis=1, ord=self.norm_metric_order)

    def _get_norm_mean_loss_of_pairs(self, positive_samples, negative_samples):
        if self.norm_metric_order is None:
            raise ValueError('Norm metric requires to define norm_metric_order')
        if self.norm_metric_margin is None:
            raise ValueError('Norm metric requires to define norm_metric_margin')
        positive_distances = tf.norm(positive_samples, axis=1, ord=self.norm_metric_order)
        negative_distances = tf.norm(negative_samples, axis=1, ord=self.norm_metric_order)
        return tf.reduce_mean(tf.nn.relu(positive_distances - negative_distances + self.norm_metric_margin))

    def get_losses_of_positive_samples(self, samples):
        if self.optimized_metric == OptimizedMetric.NORM:
            return self._get_norm_losses_of_samples(samples)
        elif self.optimized_metric == OptimizedMetric.SOFTPLUS:
            return self._get_softplus_losses_of_samples(samples)
        else:
            raise ValueError(f'Invalid optimized_metric: {self.optimized_metric}')

    def get_mean_loss_of_pairs(self, positive_samples, negative_samples):
        if self.optimized_metric == OptimizedMetric.NORM:
            return self._get_norm_mean_loss_of_pairs(positive_samples, negative_samples)
        elif self.optimized_metric == OptimizedMetric.SOFTPLUS:
            return self._get_softplus_mean_loss_of_pairs(positive_samples, negative_samples)
        else:
            raise ValueError(f'Invalid optimized_metric: {self.optimized_metric}')
