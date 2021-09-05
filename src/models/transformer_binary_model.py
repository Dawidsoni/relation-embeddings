import dataclasses
import gin.tf
import tensorflow as tf

from layers.embeddings_layers import EmbeddingsConfig
from layers.transformer_layers import StackedTransformerEncodersLayer
from models.knowledge_completion_model import EmbeddingsBasedModel


@gin.configurable(blacklist=['embeddings_config'])
class TransformerBinaryModel(EmbeddingsBasedModel):

    def __init__(self, embeddings_config: EmbeddingsConfig, use_only_edge_in_reduction_layer: bool = gin.REQUIRED):
        super().__init__(embeddings_config)
        self.transformer_layer = StackedTransformerEncodersLayer()
        self.use_only_edge_in_reduction_layer = use_only_edge_in_reduction_layer
        self.reduction_layer = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, training=None, **kwargs):
        outputs = self.embeddings_layer(inputs, training=training)
        outputs = self.transformer_layer(outputs, training=training)
        if self.use_only_edge_in_reduction_layer:
            outputs = outputs[:, :3, :]
        outputs = tf.reshape(outputs, shape=(tf.shape(outputs)[0], -1))
        outputs = self.reduction_layer(outputs, training=training)
        return outputs

    def get_config(self):
        return {
            **dataclasses.asdict(self.embeddings_config),
        }
