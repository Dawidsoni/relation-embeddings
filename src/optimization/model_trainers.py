from abc import abstractmethod
import tensorflow as tf
import gin.tf

from models.knowledge_completion_model import KnowledgeCompletionModel
from optimization.loss_objects import SamplingLossObject, SupervisedLossObject


LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


class ModelTrainer(object):

    @abstractmethod
    def train_step(self, training_samples: tf.Tensor, training_step: int):
        pass


@gin.configurable(whitelist=['negatives_reducer'])
class SamplingModelTrainer(ModelTrainer):

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: SamplingLossObject,
        learning_rate_schedule: LearningRateSchedule,
        negatives_reducer=tf.reduce_max
    ):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)
        self.negatives_reducer = negatives_reducer

    def train_step(self, training_samples: tf.Tensor, training_step: int):
        with tf.GradientTape() as gradient_tape:
            positive_inputs, array_of_negative_inputs = training_samples
            positive_outputs = self.model(positive_inputs, training=True)
            array_of_raw_losses = []
            for negative_inputs in array_of_negative_inputs:
                negative_outputs = self.model(negative_inputs, training=True)
                array_of_raw_losses.append(self.loss_object.get_losses_of_pairs(positive_outputs, negative_outputs))
            raw_losses = self.negatives_reducer(array_of_raw_losses, axis=0)
            loss_value = tf.reduce_mean(raw_losses) + self.loss_object.get_regularization_loss(self.model)
        trainable_variables = self.model.get_trainable_variables_at_training_step(training_step)
        gradients = gradient_tape.gradient(loss_value, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss_value.numpy()


class SoftmaxModelTrainer(ModelTrainer):

    def __init__(
        self,
        model: KnowledgeCompletionModel,
        loss_object: SupervisedLossObject,
        learning_rate_schedule: LearningRateSchedule
    ):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)

    def train_step(self, training_samples: tf.Tensor, training_step: int):
        inputs, true_outputs, unused_ids_of_outputs, unused_mask_indexes = training_samples
        with tf.GradientTape() as gradient_tape:
            predicted_outputs = self.model(inputs, training=True)
            loss_value = self.loss_object.get_mean_loss_of_samples(true_outputs, predicted_outputs)
        trainable_variables = self.model.get_trainable_variables_at_training_step(training_step)
        gradients = gradient_tape.gradient(loss_value, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss_value.numpy()
