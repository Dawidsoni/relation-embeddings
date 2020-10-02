import tensorflow as tf


class TrainLib(object):

    def __init__(self, model, loss_object, learning_rate_schedule, regularization_strength):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)
        self.regularization_strength = regularization_strength

    def train_step(self, positive_inputs, negative_inputs):
        with tf.GradientTape() as gradient_tape:
            positive_outputs = self.model(positive_inputs)
            negative_outputs = self.model(negative_inputs)
            raw_loss_value = self.loss_object.get_mean_loss_of_pairs(positive_outputs, negative_outputs)
            list_of_weights = [tf.reshape(weights, (-1, )) for weights in self.model.weights if len(weights.shape) > 1]
            regularization_loss = tf.norm(tf.concat(list_of_weights, axis=0), ord=2)
            loss_value = raw_loss_value + self.regularization_strength * regularization_loss
        gradients = gradient_tape.gradient(loss_value, self.model.variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
