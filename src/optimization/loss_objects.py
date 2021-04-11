from abc import ABC, abstractmethod

import tensorflow as tf
import gin.tf


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
    def get_mean_loss_of_pairs(self, positive_samples, negative_samples):
        pass


@gin.configurable
class NormLossObject(SamplingLossObject):

    def __init__(
        self, regularization_strength: float = gin.REQUIRED, order: int = gin.REQUIRED, margin: float = gin.REQUIRED
    ):
        super(NormLossObject, self).__init__(regularization_strength)
        self.order = order
        self.margin = margin

    def get_losses_of_positive_samples(self, samples):
        return tf.norm(samples, axis=1, ord=self.order)

    def get_mean_loss_of_pairs(self, positive_samples, negative_samples):
        positive_distances = tf.norm(positive_samples, axis=1, ord=self.order)
        negative_distances = tf.norm(negative_samples, axis=1, ord=self.order)
        return tf.reduce_mean(tf.nn.relu(positive_distances - negative_distances + self.margin))


@gin.configurable
class SoftplusLossObject(SamplingLossObject):

    def __init__(self, regularization_strength: float = gin.REQUIRED):
        super(SoftplusLossObject, self).__init__(regularization_strength)
        self.regularization_strength = regularization_strength

    def get_losses_of_positive_samples(self, samples):
        if samples.shape[1] != 1:
            raise ValueError('Softplus metric is incompatible with embeddings of shape greater than 1')
        return tf.reshape(tf.math.softplus(samples), shape=(-1,))

    def get_mean_loss_of_pairs(self, positive_samples, negative_samples):
        positive_losses = self.get_losses_of_positive_samples(positive_samples)
        negative_losses = self.get_losses_of_positive_samples(-negative_samples)
        return tf.reduce_mean(tf.concat([positive_losses, negative_losses], axis=0)) / 2.0


class SupervisedLossObject(LossObject):

    def __init__(self, regularization_strength: float = gin.REQUIRED):
        super(SupervisedLossObject, self).__init__(regularization_strength)
        self.regularization_strength = regularization_strength

    @abstractmethod
    def get_mean_loss_of_samples(self, true_labels, soft_predictions):
        pass  # TODO: implement this method
