import tensorflow as tf

from conv_base_model import ConvBaseModel


class TranseModel(ConvBaseModel):

    def __init__(self, entities_count, relations_count, embedding_dimension, include_reduce_dim_layer):
        super().__init__(
            entities_count, relations_count, embedding_dimension, include_reduce_dim_layer=include_reduce_dim_layer,
            trainable_embeddings=True
        )
        kernel_weights = tf.constant_initializer([[[[1]], [[1]], [[-1]]]])
        self._conv_layers = [
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 3), kernel_initializer=kernel_weights, trainable=False)
        ]

    @property
    def conv_layers(self):
        return self._conv_layers
