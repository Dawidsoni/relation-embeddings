import tensorflow as tf
import tensorflow_addons as tfa
import gin.tf


def get_embeddings_initializer():
    return tf.keras.initializers.TruncatedNormal(stddev=0.02)


@gin.configurable
def get_parameters_initializer(use_glorot=False):
    if use_glorot:
        return tf.keras.initializers.GlorotNormal()
    return tf.keras.initializers.TruncatedNormal(stddev=0.02)


def get_activation():
    return tf.keras.activations.gelu


@gin.configurable(whitelist=["weight_decay"])
def create_optimizer(learning_rate_schedule, weight_decay=0.01):
    # return tf.optimizers.Adam(learning_rate_schedule)
    return tfa.optimizers.AdamW(weight_decay, learning_rate_schedule)
