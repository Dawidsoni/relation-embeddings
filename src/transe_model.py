import tensorflow as tf
import gin.tf

from conv_base_model import ConvBaseModel, DataConfig, ModelConfig


@gin.configurable(denylist=['data_config'])
class TranseModel(ConvBaseModel):

    def __init__(self, data_config: DataConfig, model_config: ModelConfig = gin.REQUIRED):
        super().__init__(data_config, model_config)
        kernel_weights = tf.constant_initializer([[[[1]], [[1]], [[-1]]]])
        self._conv_layers = [
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 3), kernel_initializer=kernel_weights, trainable=False)
        ]

    @property
    def conv_layers(self):
        return self._conv_layers
