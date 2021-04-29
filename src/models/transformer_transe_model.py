import dataclasses
import gin.tf

from layers.embeddings_layers import EmbeddingsConfig
from layers.transformer_layers import StackedTransformerEncodersLayer
from models.conv_base_model import ConvModelConfig
from models.transe_model import TranseModel


@gin.configurable(blacklist=['embeddings_config'])
class TransformerTranseModel(TranseModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: ConvModelConfig = gin.REQUIRED):
        super().__init__(embeddings_config, model_config)
        self.transformer_layer = StackedTransformerEncodersLayer()

    def call(self, inputs, training=None, **kwargs):
        outputs = self.embeddings_layer(inputs, training=training)
        outputs = self.transformer_layer(outputs, training=training)
        return self._transform_and_rate_embeddings(outputs, training=training)

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
        }
