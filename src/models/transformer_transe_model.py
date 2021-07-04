import dataclasses
import gin.tf
import tensorflow as tf

from layers.embeddings_layers import EmbeddingsConfig
from layers.transformer_layers import StackedTransformerEncodersLayer
from models.conv_base_model import ConvModelConfig
from models.transe_model import TranseModel


@gin.configurable(blacklist=['embeddings_config'])
class TransformerTranseModel(TranseModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, model_config: ConvModelConfig = gin.REQUIRED):
        parent_model_config = dataclasses.replace(model_config, normalize_embeddings=False)
        super().__init__(embeddings_config, parent_model_config)
        self.transformer_layer = StackedTransformerEncodersLayer()
        self.normalize_embeddings = model_config.normalize_embeddings

    def call(self, inputs, training=None, **kwargs):
        outputs = self.embeddings_layer(inputs, training=training)
        if self.normalize_embeddings:
            outputs = tf.math.l2_normalize(outputs, axis=2)
        outputs = self.transformer_layer(outputs, training=training)
        edge_outputs = outputs[:, :3, :]
        return self._transform_and_rate_embeddings(edge_outputs, training=training)

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
        }
