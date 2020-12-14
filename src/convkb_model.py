import tensorflow as tf
import gin.tf
from dataclasses import dataclass
from typing import Sequence, Union

from conv_base_model import ConvBaseModel, DataConfig, ModelConfig


@gin.configurable
@dataclass
class ConvolutionsConfig:
    filter_heights: Sequence[int]
    filters_count_per_height: int
    activation: Union[str, tf.keras.layers.Activation]


@gin.configurable(blacklist=['data_config'])
class ConvKBModel(ConvBaseModel):

    def __init__(
        self, data_config: DataConfig, model_config: ModelConfig = gin.REQUIRED,
        convolutions_config: ConvolutionsConfig = gin.REQUIRED
    ):
        super(ConvKBModel, self).__init__(data_config, model_config)
        filters_count_per_height = convolutions_config.filters_count_per_height
        conv_activation = convolutions_config.activation
        self._conv_layers = [
            tf.keras.layers.Conv2D(filters_count_per_height, kernel_size=(filter_height, 3), activation=conv_activation)
            for filter_height in convolutions_config.filter_heights
        ]

    @property
    def conv_layers(self):
        return self._conv_layers
