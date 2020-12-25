import tensorflow as tf
import gin.tf
from dataclasses import dataclass
from typing import Sequence, Union
import functools

from conv_base_model import ConvBaseModel, DataConfig, ModelConfig


@gin.configurable
@dataclass
class ConvolutionsConfig:
    filter_heights: Sequence[int]
    filters_count_per_height: int
    activation: Union[str, tf.keras.layers.Activation]
    use_constant_initialization: bool


class ConvolutionWeightsInitializer(tf.keras.initializers.Initializer):

    def __init__(self, use_constant_initialization: bool):
        self.use_constant_initialization = use_constant_initialization
        self.normal_initializer = tf.keras.initializers.TruncatedNormal()
        self.initialization_constant = tf.constant([[[[0.1]], [[0.1]], [[-0.1]]]])

    def __call__(self, shape, **kwargs):
        if len(shape) != 4:
            raise ValueError(f"Expected a Tensor of rank 4, got {tf.rank(shape)}")
        if self.use_constant_initialization:
            multiples = (shape[0], 1, shape[2], shape[3])
            return tf.tile(self.initialization_constant, multiples)
        else:
            return self.normal_initializer(shape)

    def get_config(self):
        return {"use_constant_initialization": self.use_constant_initialization}


@gin.configurable(blacklist=['data_config'])
class ConvKBModel(ConvBaseModel):

    def __init__(
        self, data_config: DataConfig, model_config: ModelConfig = gin.REQUIRED,
        convolutions_config: ConvolutionsConfig = gin.REQUIRED
    ):
        super(ConvKBModel, self).__init__(data_config, model_config)
        filters_count_per_height = convolutions_config.filters_count_per_height
        conv_activation = convolutions_config.activation
        conv_layer_func = functools.partial(
            tf.keras.layers.Conv2D,
            filters=filters_count_per_height,
            activation=conv_activation,
            kernel_initializer=ConvolutionWeightsInitializer(convolutions_config.use_constant_initialization)
        )
        self._conv_layers = [
            conv_layer_func(kernel_size=(filter_height, 3)) for filter_height in convolutions_config.filter_heights
        ]

    @property
    def conv_layers(self):
        return self._conv_layers
