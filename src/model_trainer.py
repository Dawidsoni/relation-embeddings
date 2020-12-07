import tensorflow as tf


class ModelTrainer(object):

    def __init__(self, model, loss_object, learning_rate_schedule, regularization_strength):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)
        self.regularization_strength = regularization_strength

    def get_regularization_loss(self):
        list_of_weights = [weights for weights in self.model.trainable_variables if len(weights.shape) > 1]
        losses = [tf.reshape(tf.norm(weights, axis=-1), (-1,)) for weights in list_of_weights]
        return self.regularization_strength * tf.reduce_mean(tf.concat(losses, axis=0))

    def train_step(self, positive_inputs, negative_inputs):
        with tf.GradientTape() as gradient_tape:
            positive_outputs = self.model(positive_inputs)
            negative_outputs = self.model(negative_inputs)
            raw_loss_value = self.loss_object.get_mean_loss_of_pairs(positive_outputs, negative_outputs)
            loss_value = raw_loss_value + self.get_regularization_loss()
        gradients = gradient_tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
