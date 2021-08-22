import tensorflow as tf
import tensorflow_addons as tfa
import gin.tf


@gin.configurable
def get_embeddings_initializer(stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


@gin.configurable
def get_parameters_initializer(stddev=0.02, use_glorot=False):
    if use_glorot:
        return tf.keras.initializers.GlorotNormal()
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def get_activation():
    return tf.keras.activations.gelu


@gin.configurable(whitelist=["weight_decay"])
def create_optimizer(learning_rate_schedule, weight_decay=0.0):
    return tfa.optimizers.AdamW(weight_decay, learning_rate_schedule)
