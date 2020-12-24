import tensorflow as tf


class ModelTrainer(object):

    def __init__(self, model, loss_object, learning_rate_schedule):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)

    def train_step(self, positive_inputs, negative_inputs):
        with tf.GradientTape() as gradient_tape:
            positive_outputs = self.model(positive_inputs, training=True)
            negative_outputs = self.model(negative_inputs, training=True)
            raw_loss_value = self.loss_object.get_mean_loss_of_pairs(positive_outputs, negative_outputs)
            loss_value = raw_loss_value + self.loss_object.get_regularization_loss(self.model)
        gradients = gradient_tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

