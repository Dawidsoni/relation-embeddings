import dataclasses
import tensorflow as tf
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from models.conv_base_model import ConvBaseModel, ConvModelConfig


@gin.configurable(blacklist=['embeddings_config'])
class TranseModel(ConvBaseModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: ConvModelConfig = gin.REQUIRED):
        super().__init__(embeddings_config, model_config)
        kernel_weights = tf.constant_initializer([[[[1]], [[1]], [[-1]]]])
        self._conv_layers = [
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 3), kernel_initializer=kernel_weights, trainable=False)
        ]

    @property
    def conv_layers(self):
        return self._conv_layers

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
            **dataclasses.asdict(self.model_config),
        }
