from abc import ABC, abstractmethod

import tensorflow as tf
import gin.tf


@gin.configurable
class LossObject(ABC):

    def __init__(self, regularization_strength: float = gin.REQUIRED):
        self.regularization_strength = regularization_strength

    def get_regularization_loss(self, model):
        list_of_weights = [weights for weights in model.trainable_variables if len(weights.shape) > 1]
        losses = [tf.reshape(tf.norm(weights, axis=-1), (-1,)) for weights in list_of_weights]
        return self.regularization_strength * tf.reduce_mean(tf.concat(losses, axis=0))


class SamplingLossObject(LossObject):

    @abstractmethod
    def get_losses_of_positive_samples(self, samples):
        pass

    @abstractmethod
    def get_losses_of_pairs(self, positive_samples, negative_samples):
        pass

    def get_mean_loss_of_pairs(self, positive_samples, negative_samples):
        return tf.reduce_mean(self.get_losses_of_pairs(positive_samples, negative_samples))


@gin.configurable
class NormLossObject(SamplingLossObject):

    def __init__(self, order: int = gin.REQUIRED, margin: float = gin.REQUIRED):
        super(NormLossObject, self).__init__()
        self.order = order
        self.margin = margin

    def get_losses_of_positive_samples(self, samples):
        return tf.norm(samples, axis=1, ord=self.order)

    def get_losses_of_pairs(self, positive_samples, negative_samples):
        positive_distances = tf.norm(positive_samples, axis=1, ord=self.order)
        negative_distances = tf.norm(negative_samples, axis=1, ord=self.order)
        return tf.nn.relu(positive_distances - negative_distances + self.margin)


@gin.configurable
class SoftplusLossObject(SamplingLossObject):

    def get_losses_of_positive_samples(self, samples):
        if samples.shape[1] != 1:
            raise ValueError('Softplus metric is incompatible with embeddings of shape greater than 1')
        return tf.reshape(tf.math.softplus(samples), shape=(-1,))

    def get_losses_of_pairs(self, positive_samples, negative_samples):
        positive_losses = self.get_losses_of_positive_samples(positive_samples)
        negative_losses = self.get_losses_of_positive_samples(-negative_samples)
        return tf.concat([positive_losses, negative_losses], axis=0) / 2.0


@gin.configurable
class BinaryCrossEntropyLossObject(SamplingLossObject):

    def __init__(self, label_smoothing):
        super(BinaryCrossEntropyLossObject, self).__init__()
        self.loss_function = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE, label_smoothing=label_smoothing
        )

    def _get_losses_of_samples(self, labels, samples):
        if samples.shape[1] != 1:
            raise ValueError('BinaryCrossEntropy metric is incompatible with embeddings of shape greater than 1')
        return self.loss_function(y_true=labels, y_pred=samples)

    def get_losses_of_positive_samples(self, samples):
        return self._get_losses_of_samples(labels=tf.ones_like(samples, dtype=tf.int32), samples=samples)

    def get_losses_of_pairs(self, positive_samples, negative_samples):
        positive_losses = self._get_losses_of_samples(
            labels=tf.ones_like(positive_samples, dtype=tf.int32), samples=positive_samples
        )
        negative_losses = self._get_losses_of_samples(
            labels=tf.zeros_like(negative_samples, dtype=tf.int32), samples=negative_samples
        )
        return (positive_losses + negative_losses) / 2.0


class SupervisedLossObject(LossObject):

    @abstractmethod
    def get_mean_loss_of_samples(self, true_labels, predictions):
        pass

    @abstractmethod
    def get_losses_of_samples(self, true_labels, predictions):
        pass


@gin.configurable
class CrossEntropyLossObject(SupervisedLossObject):

    def __init__(self, label_smoothing):
        super(CrossEntropyLossObject, self).__init__()
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=label_smoothing
        )

    def get_mean_loss_of_samples(self, true_labels, predictions):
        return self.loss_function(true_labels, predictions)

    def get_losses_of_samples(self, true_labels, predictions):
        return 1 - predictions
